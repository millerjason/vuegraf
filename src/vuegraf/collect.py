# Copyright (c) Jason Ertel (jertel).
# This file is part of the Vuegraf project and is made available under the MIT License.

# Contains logic relating to collection of data usage from Emporia cloud

import datetime
from dataclasses import dataclass
import logging
import time
from typing import Union

from pyemvue.enums import Scale, Unit

from vuegraf.cache import (
    bootstrapManifestFromCsvs,
    getCacheCsvPath,
    isCacheEnabled,
    loadManifest,
    loadMirroredKeys,
    pointKey,
    restoreCacheToDatabase,
    saveManifest,
    subtractCoverage,
    writeCsvPoints,
)
from vuegraf.config import getConfigValue
from vuegraf.device import lookupDeviceName, lookupChannelName
from vuegraf.destination import getLastDBTimeStamp, getTags, writeDataPoints
from vuegraf.hierarchy import applyBalances, registerImplicitEdge
from vuegraf.time import calculateHistoryTimeRange, convertToLocalDayInUTC


logger = logging.getLogger('vuegraf.data')


# How long to remember that a (device, channel) returned no minute-history data,
# before attempting the 7-day rewind backfill again. See getMinuteBackfillSkipCache().
MINUTE_BACKFILL_SKIP_TTL_SEC = 3600  # 1 hour


def getMinuteBackfillSkipCache(config):
    """Negative cache for the minute-history backfill loop in extractDataPoints.

    Some Emporia hardware revisions persistently return all-None from
    get_chart_usage(scale=MINUTE) for a device's synthesized parent channel
    (e.g. '1,2,3'), even though hourly/daily aggregates and the live-sample
    endpoint return real values for the same channel. Without a cache, when
    InfluxDB has no minute-tagged record for such a channel, getLastDBTimeStamp
    re-anchors at now - 7 days every cycle, and the backfill while-loop here
    walks the full 7-day window in 12-hour chunks — fetching nothing and
    writing nothing — on every single 60s cycle, indefinitely.

    This cache records '(deviceName, chanName) -> expiry_epoch' for channels
    where a backfill window completed without writing any points. Subsequent
    cycles short-circuit to the simple current-sample minute point (same path
    excluded channels already take) until the cache entry expires.

    The cache is bound to the config dict (one cache per Vuegraf process). It
    stays empty for channels that have data — zero overhead for working
    installations. Restarting Vuegraf clears it, giving the upstream API a
    fresh attempt.
    """
    return config.setdefault('_minuteBackfillSkipCache', {})


@dataclass
class Point:
    """Container for timestamped device readings from Vue.

    These can be repacked by Influx and MQTT when writing to those engines.
    """
    accountName: str
    deviceName: str  # aka station; the Vue, smart plug, etc
    chanName: str  # for example, each circuit or the total/balance
    usageWatts: float
    timestamp: datetime.datetime  # zone aware, UTC
    detailed: Union[bool, str]  # 'Minutes', 'Days', etc or False


def extractDataPoints(config, account, device, stopTimeUTC, collectDetails, usageDataPoints: list[Point],
                      detailedStartTimeUTC, pointType=None, historyStartTimeUTC=None, historyEndTimeUTC=None):
    """Unpacks Vue API usage data from a fetched device. Module use only.

    Modifies usageDataPoints in place, appending Point objects.
    """
    accountName = account['name']
    detailedDataEnabled = getConfigValue(config, 'detailedDataEnabled')
    detailedSecondsEnabled = detailedDataEnabled and getConfigValue(config, 'detailedDataSecondsEnabled')
    _, tagValue_second, tagValue_minute, tagValue_hour, tagValue_day = getTags(config)
    excludedDetailChannelNumbers = ['Balance', 'TotalUsage']
    minutesInAnHour = 60
    secondsInAMinute = 60
    wattsInAKw = 1000
    deviceName = lookupDeviceName(account, device.device_gid)

    for chanNum, chan in device.channels.items():
        if chan.nested_devices:
            # Emporia reports its own nesting only on usage responses, so this is where the
            # implicit edges become visible. Registering them here folds them into the same
            # tree as the config-declared ones. No-op unless the account opted in.
            nestedParentChanName = lookupChannelName(account, chan)
            for gid, nestedDevice in chan.nested_devices.items():
                registerImplicitEdge(config, accountName, lookupDeviceName(account, gid), nestedParentChanName)
                extractDataPoints(config, account, nestedDevice, stopTimeUTC, collectDetails, usageDataPoints,
                                  detailedStartTimeUTC, pointType, historyStartTimeUTC, historyEndTimeUTC)

        chanName = lookupChannelName(account, chan)
        kwhUsage = chan.usage
        if kwhUsage is not None:
            if pointType is None:
                # Negative-cache check: if a prior backfill window for this
                # (device, channel) completed without writing any minute points,
                # short-circuit to the simple current-sample branch and skip the
                # 7-day-rewind getLastDBTimeStamp + while-loop entirely. See
                # getMinuteBackfillSkipCache for the rationale.
                skipCache = getMinuteBackfillSkipCache(config)
                cacheKey = (deviceName, chanName)
                skipExpiry = skipCache.get(cacheKey)
                if skipExpiry is not None and skipExpiry > time.time():
                    minuteHistoryStartTime, stopTimeMin, minuteHistoryEnabled = (stopTimeUTC, stopTimeUTC, False)
                else:
                    # Collect previous minute averages
                    minuteHistoryStartTime, stopTimeMin, minuteHistoryEnabled = getLastDBTimeStamp(config, deviceName,
                                                                                                   chanName, tagValue_minute,
                                                                                                   stopTimeUTC, stopTimeUTC, False)
                if not minuteHistoryEnabled or chanNum in excludedDetailChannelNumbers:
                    watts = float(minutesInAnHour * wattsInAKw) * kwhUsage
                    timestamp = stopTimeUTC.replace(second=0)
                    usageDataPoints.append(Point(accountName, deviceName, chanName, watts, timestamp, tagValue_minute))
                elif chanNum not in excludedDetailChannelNumbers and historyStartTimeUTC is None:
                    # Still missing recent minute history, attempt to collect in batches of 12 hours
                    pointsBeforeBackfill = len(usageDataPoints)
                    noDataFlag = True
                    while noDataFlag:
                        # Collect minutes history (if neccessary, never during history collection)
                        logger.info('Get minute details; device="{}"; start="{}"; stop="{}"'.format(chanName,
                                    minuteHistoryStartTime, stopTimeMin))
                        usage, usage_start_time = account['vue'].get_chart_usage(chan, minuteHistoryStartTime, stopTimeMin,
                                                                                 scale=Scale.MINUTE.value, unit=Unit.KWH.value)
                        usage_start_time = usage_start_time.replace(second=0, microsecond=0)
                        index = 0
                        for kwhUsage in usage:
                            if kwhUsage is None:
                                index += 1
                                continue
                            noDataFlag = False  # Got at least one datapoint.  Set boolean value so we don't loop back
                            timestamp = usage_start_time + datetime.timedelta(minutes=index)
                            watts = float(minutesInAnHour * wattsInAKw) * kwhUsage
                            usageDataPoints.append(
                                Point(
                                    accountName, deviceName, chanName, watts,
                                    timestamp, tagValue_minute
                                )
                            )
                            index += 1
                        if noDataFlag:
                            # Opps!  No data points found for the time interval in question ('None' returned for ALL values)
                            # Move up the time interval to the next "batch" timeframe
                            if stopTimeMin < stopTimeUTC.replace(second=0, microsecond=0):
                                currentIntervalSeconds = int((stopTimeMin - minuteHistoryStartTime).total_seconds())
                                minuteHistoryStartTime = minuteHistoryStartTime + datetime.timedelta(seconds=currentIntervalSeconds)
                                # Make sure we don't go beyond the global stopTimeUTC
                                minuteHistoryStartTime = min(minuteHistoryStartTime, stopTimeUTC.replace(second=0, microsecond=0))
                                stopTimeMin = stopTimeMin + datetime.timedelta(seconds=currentIntervalSeconds)
                                # Make sure we don't go beyond the global stopTimeUTC
                                stopTimeMin = min(stopTimeMin, stopTimeUTC.replace(second=0, microsecond=0))
                            else:  # Time to break out of the loop; looks like the device in question is offline
                                noDataFlag = False
                        minuteHistoryEnabled = False
                    if len(usageDataPoints) == pointsBeforeBackfill:
                        # The backfill window completed without writing any minute
                        # points -- the upstream API returned all-None across the
                        # entire 7-day rewind for this (device, channel). Cache the
                        # negative result so subsequent cycles skip the rewind.
                        skipCache[cacheKey] = time.time() + MINUTE_BACKFILL_SKIP_TTL_SEC
                        logger.info(
                            'No historical minute data for device="%s"; suppressing '
                            'minute backfill for %ds (cache).',
                            chanName, MINUTE_BACKFILL_SKIP_TTL_SEC,
                        )
            elif pointType == tagValue_day:
                # Collect previous day averages
                watts = kwhUsage * wattsInAKw
                timestamp = convertToLocalDayInUTC(config, historyStartTimeUTC)
                usageDataPoints.append(Point(accountName, deviceName, chanName, watts, timestamp, pointType))
            elif pointType == tagValue_hour:
                # Collect previous hour averages
                watts = kwhUsage * wattsInAKw
                timestamp = historyStartTimeUTC
                usageDataPoints.append(Point(accountName, deviceName, chanName, watts, timestamp, pointType))

        if chanNum in excludedDetailChannelNumbers:
            continue

        if collectDetails and detailedSecondsEnabled and historyStartTimeUTC is None:
            # Collect seconds (once per hour, never during history collection)
            secHistoryStartTime, stopTimeSec, secondHistoryEnabled = getLastDBTimeStamp(config, deviceName, chanName, tagValue_second,
                                                                                        detailedStartTimeUTC, stopTimeUTC,
                                                                                        detailedSecondsEnabled)
            logger.debug('Get second details; device="{}"; start="{}"; stop="{}"'.format(chanName, secHistoryStartTime, stopTimeSec))
            usage, usageStartTimeUTC = account['vue'].get_chart_usage(chan, secHistoryStartTime, stopTimeSec, scale=Scale.SECOND.value,
                                                                      unit=Unit.KWH.value)
            usageStartTimeUTC = usageStartTimeUTC.replace(microsecond=0)
            index = 0
            for kwhUsage in usage:
                if kwhUsage is None:
                    index += 1
                    continue
                timestamp = usageStartTimeUTC + datetime.timedelta(seconds=index)
                watts = float(secondsInAMinute * minutesInAnHour * wattsInAKw) * kwhUsage
                usageDataPoints.append(Point(accountName, deviceName, chanName, watts, timestamp, tagValue_second))
                index += 1

        # Fetches historical Hour & Day data
        if historyStartTimeUTC is not None and historyEndTimeUTC is not None:
            logger.debug('Get historic details; device="{}"; start="{}"; stop="{}"'.format(chanName, historyStartTimeUTC,
                                                                                           historyEndTimeUTC))

            # Collect historical hour averages
            usage, usageStartTimeUTC = account['vue'].get_chart_usage(chan, historyStartTimeUTC, historyEndTimeUTC,
                                                                      scale=Scale.HOUR.value, unit=Unit.KWH.value)
            usageStartTimeUTC = usageStartTimeUTC.replace(minute=0, second=0, microsecond=0)
            index = 0
            for kwhUsage in usage:
                if kwhUsage is None:
                    index += 1
                    continue
                timestamp = usageStartTimeUTC + datetime.timedelta(hours=index)
                watts = kwhUsage * wattsInAKw
                usageDataPoints.append(Point(accountName, deviceName, chanName,
                                       watts, timestamp, tagValue_hour))
                index += 1

            # Collect historical day averages
            usage, usageStartTimeUTC = account['vue'].get_chart_usage(chan, historyStartTimeUTC, historyEndTimeUTC,
                                                                      scale=Scale.DAY.value, unit=Unit.KWH.value)
            index = 0
            for kwhUsage in usage:
                if kwhUsage is None:
                    index += 1
                    continue

                # Advance the day by 6 hours + the current day index. The 6 hours shifts time away from common DST threshold hours
                # to avoid DST issues. Note that historyStartTimeUTC will be set to midnight by the caller, thus
                # usageStartTimeUTC, returned by Emporia, will be midnight as well.
                timestamp = convertToLocalDayInUTC(config, usageStartTimeUTC + datetime.timedelta(hours=6, days=index))

                watts = kwhUsage * wattsInAKw
                usageDataPoints.append(Point(accountName, deviceName, chanName, watts, timestamp, tagValue_day))
                index += 1


def collectUsage(config, account, startTimeUTC, stopTimeUTC, collectDetails, usageDataPoints: list[Point], detailedStartTimeUTC, scale):
    """Module entrypoint. Fetch Vue data and unpack it into points.

    The usageDataPoints list is modified in place, appending Points.
    """
    _, _, _, tagValue_hour, tagValue_day = getTags(config)
    if scale == Scale.HOUR.value:
        pointType = tagValue_hour
    elif scale == Scale.DAY.value:
        pointType = tagValue_day
    else:
        pointType = None

    logger.debug('Collecting data from Emporia; Scale={}; startTimeUTC={}; stopTimeUTC={}'.format(scale, startTimeUTC, stopTimeUTC))

    deviceGids = list(account['deviceIdMap'].keys())
    usages = account['vue'].get_device_list_usage(deviceGids, stopTimeUTC, scale=scale, unit=Unit.KWH.value)
    if usages is not None:
        for gid, device in usages.items():
            extractDataPoints(config, account, device, stopTimeUTC, collectDetails,
                              usageDataPoints, detailedStartTimeUTC, pointType, startTimeUTC)


def writeHistoryBatchCsv(csvPath, points: list[Point], mirroredKeys=frozenset()):
    """Appends a batch of history points to the CSV cache, for recovery/inspection.

    Points already mirrored by the running daemon are skipped, so the shared
    account file does not accumulate a second copy of them. csvPath is None when
    the CSV cache is disabled, in which case history goes only to the database.
    """
    if csvPath is None:
        return
    fresh = [pt for pt in points if pointKey(pt) not in mirroredKeys]
    if fresh:
        writeCsvPoints(csvPath, fresh)


def fetchHistoryRange(config, account, usages, rangeStartUTC, rangeEndUTC, csvPath, usageDataPoints, pauseEvent,
                      mirroredKeys=frozenset()):
    """Fetch history from the Emporia API for a single [start, stop] range.

    Batches the range into <=20-day increments, persisting each batch to the CSV
    cache (first, so a failed database write still leaves a recoverable file) and
    to the database, then clearing it from memory. usageDataPoints is reused as the
    per-batch buffer and is left empty on return.

    Returns (pointsWritten, earliestDataUTC, lastCompletedStopUTC, aborted) where
    lastCompletedStopUTC is how far coverage may be safely advanced (== the last
    batch's stop, or rangeStartUTC if nothing completed).
    """
    totalDays = max(1, (rangeEndUTC - rangeStartUTC).days)
    totalBatches = max(1, -(-totalDays // 20))  # ceil; matches 20-day increments in calculateHistoryTimeRange
    logger.info('Fetching uncovered history from API; start={}; stop={}; batches={}'.format(
                rangeStartUTC, rangeEndUTC, totalBatches))

    pointsWritten = 0
    earliestDataUTC = None
    lastCompletedStopUTC = rangeStartUTC
    aborted = False
    batchCounter = 0
    while True:
        # Anchor the 20-day windows at rangeStartUTC and clamp the stop to rangeEndUTC.
        incrementStartTimeUTC, incrementEndTimeUTC = calculateHistoryTimeRange(config, rangeEndUTC, rangeStartUTC,
                                                                               batchCounter)
        if incrementStartTimeUTC >= rangeEndUTC:
            break

        logger.info('History batch {}/{}; start={}; stop={}'.format(
                    batchCounter + 1, totalBatches, incrementStartTimeUTC, incrementEndTimeUTC))
        for gid, device in usages.items():
            extractDataPoints(config, account, device, rangeEndUTC, False, usageDataPoints, None,
                              'History', incrementStartTimeUTC, incrementEndTimeUTC)

        # A batch already holds every device for its window, so the balance pass runs per
        # batch with no cross-batch state. Appending here, before the writes below, is what
        # gets the Net Balance series into both the CSV mirror and InfluxDB.
        applyBalances(config, usageDataPoints)

        if usageDataPoints:
            batchEarliest = min(pt.timestamp for pt in usageDataPoints)
            if earliestDataUTC is None or batchEarliest < earliestDataUTC:
                earliestDataUTC = batchEarliest

        batchPoints = len(usageDataPoints)
        pointsWritten += batchPoints
        writeHistoryBatchCsv(csvPath, usageDataPoints, mirroredKeys)
        writeDataPoints(config, usageDataPoints)
        del usageDataPoints[:]
        lastCompletedStopUTC = incrementEndTimeUTC
        logger.info('History batch {}/{} complete; batchPoints={}; totalPoints={}'.format(
                    batchCounter + 1, totalBatches, batchPoints, pointsWritten))

        batchCounter = batchCounter + 1

        if pauseEvent.wait(5):
            logging.info("Aborting history collection due to pause interruption")
            aborted = True
            break

    return pointsWritten, earliestDataUTC, lastCompletedStopUTC, aborted


def collectHistoryUsage(config, account, startTimeUTC, stopTimeUTC, usageDataPoints: list[Point], pauseEvent):
    """Module entrypoint. Populates the database with historic Vue data.

    When the CSV cache is disabled (the default) the requested range is simply
    fetched from the Emporia API in batches and written to the database.

    When 'csvCacheEnabled' is set, the backfill CSV files act as a passthrough
    cache in front of the Emporia history API (see vuegraf.cache):
      1. Any part of the requested range already probed (per the cache manifest)
         is restored straight from CSV into the database -- no API calls. Restore can
         be skipped with --skiprestore when the database is known to be populated.
      2. Only the uncovered portion(s) of the requested range are fetched from
         the API (the "true-up"), appended to the cache, and written to the database.
      3. The manifest coverage is extended to include everything probed --
         including windows that returned no data -- so future runs never re-probe
         the empty pre-install era, regardless of the --historydays range used.
    """
    # Grab base usage data for later use in history collection
    deviceGids = list(account['deviceIdMap'].keys())
    usages = account['vue'].get_device_list_usage(deviceGids, stopTimeUTC, scale=Scale.MINUTE.value, unit=Unit.KWH.value)

    cacheEnabled = isCacheEnabled(config)

    # Resolve current cache coverage (bootstrapping from existing CSVs the first time).
    manifest = None
    if cacheEnabled:
        manifest = loadManifest(config, account)
    if cacheEnabled and manifest is None:
        bootstrapped = bootstrapManifestFromCsvs(config, account)
        if bootstrapped is not None:
            covStart, covEnd, earliestData = bootstrapped
            manifest = {'coveredStartUTC': covStart, 'coveredEndUTC': covEnd, 'earliestDataUTC': earliestData}
            logger.info('Bootstrapped history cache from existing CSVs; data={}..{}'.format(earliestData, covEnd))

    covStartUTC = manifest['coveredStartUTC'] if manifest else None
    covEndUTC = manifest['coveredEndUTC'] if manifest else None
    earliestDataUTC = manifest['earliestDataUTC'] if manifest else None

    logger.info('History collection started; requested={}..{}; cachedCoverage={}..{}'.format(
                startTimeUTC, stopTimeUTC, covStartUTC, covEndUTC))

    # 1) Serve the already-cached portion of the request from CSV into the database (no API calls).
    skipRestore = getattr(config.get('args'), 'skiprestore', False)
    if manifest and not skipRestore:
        restoreStartUTC = max(startTimeUTC, covStartUTC)
        restoreStopUTC = min(stopTimeUTC, covEndUTC)
        if restoreStartUTC < restoreStopUTC:
            logger.info('Restoring cached history from CSV; start={}; stop={}'.format(restoreStartUTC, restoreStopUTC))
            restored = restoreCacheToDatabase(config, account, restoreStartUTC, restoreStopUTC)
            logger.info('Restored {} cached points from CSV (no API calls)'.format(restored))
    elif manifest and skipRestore:
        logger.info('Skipping CSV restore (--skiprestore); cached range assumed already present in the database')

    # 2) Fetch only the uncovered portion(s) from the API and append them to the cache.
    gaps = subtractCoverage(startTimeUTC, stopTimeUTC, covStartUTC, covEndUTC)
    if not gaps:
        logger.info('History cache fully covers the requested range; no API calls needed')

    csvPath = getCacheCsvPath(config, account) if cacheEnabled else None
    coveredPieces = []
    if covStartUTC is not None and covEndUTC is not None:
        coveredPieces.append((covStartUTC, covEndUTC))

    # Readings the daemon already mirrored into the shared file, so the batches
    # below do not append a second copy of them.
    mirroredKeys = loadMirroredKeys(config, account, covEndUTC) if cacheEnabled else frozenset()

    for gapStartUTC, gapEndUTC in gaps:
        _, dataMinUTC, lastStopUTC, aborted = fetchHistoryRange(
            config, account, usages, gapStartUTC, gapEndUTC, csvPath, usageDataPoints, pauseEvent, mirroredKeys)
        if dataMinUTC is not None and (earliestDataUTC is None or dataMinUTC < earliestDataUTC):
            earliestDataUTC = dataMinUTC
        # Only claim coverage for what actually completed (matters if aborted mid-gap).
        if lastStopUTC > gapStartUTC:
            coveredPieces.append((gapStartUTC, lastStopUTC))
        if aborted:
            break

    # 3) Persist the merged coverage. Pieces are contiguous by construction, so the
    #    union is simply [min start, max stop]. Empty (probed-but-no-data) windows are
    #    included, so they are never re-probed on future runs.
    if cacheEnabled and coveredPieces:
        newCovStartUTC = min(s for s, _ in coveredPieces)
        newCovEndUTC = max(e for _, e in coveredPieces)
        saveManifest(config, account, newCovStartUTC, newCovEndUTC, earliestDataUTC)

    logger.info('History collection finished; csv={}'.format(csvPath))
