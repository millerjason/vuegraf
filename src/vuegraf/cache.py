# Copyright (c) Jason Ertel (jertel).
# This file is part of the Vuegraf project and is made available under the MIT License.

# CSV mirror of collected data, and a passthrough cache for historical backfill.
#
# The feature is opt-in via the 'csvCacheEnabled' config setting and writes into
# the 'csvCacheDir' directory. Each account has exactly one data file,
# <account>.csv, which is appended to by both collection paths:
#
#   * historical backfill, one append per fetched batch, and
#   * normal operation, one append per hourly/daily rollup as it is collected.
#
# Keeping a single file means the CSV is a self-contained, lower-resolution
# mirror of the database that can be loaded directly (pandas, DuckDB, a
# spreadsheet) without stitching files together. Per-minute points are not
# mirrored: they are orders of magnitude more voluminous and are already durable
# in the database.
#
# A small JSON manifest per account records which time range has already been
# *probed* (coveredStartUTC..coveredEndUTC) -- including windows that returned no
# data -- plus the earliest timestamp that actually had data (earliestDataUTC).
# This lets a re-run:
#   * recover already-collected history from CSV into the database (no API calls),
#   * skip API calls for any window already probed (even empty pre-install eras),
#   * make API calls only for the uncovered portion(s) of the requested range.
#
# Coverage is tracked as absolute UTC instants, not batch offsets, so it remains
# correct when the requested --historydays range changes between runs.
#
# Because both paths append to one file, a backfill covering a period the daemon
# already mirrored would write those points twice. loadMirroredKeys() prevents
# that at the source: before fetching, the keys already present at or after the
# current coverage end are read back, and matching points are simply not
# re-appended. That window is bounded by design -- rollups only ever write
# near-now points, and anything older came from a backfill, which the coverage
# calculation already excludes -- so the key set stays small no matter how much
# history the file holds.

import csv
import datetime
import json
import logging
import os
import re

from vuegraf.config import getConfigValue
from vuegraf.destination import writeDataPoints


logger = logging.getLogger('vuegraf.cache')

CSV_HEADER = ['account_name', 'device_name', 'channel_name', 'usage_watts', 'timestamp_utc', 'detailed']

# Number of cached points to buffer before flushing to the database during a restore.
RESTORE_CHUNK = 50000


def isCacheEnabled(config):
    """True when the CSV cache/mirror is turned on in the configuration."""
    return getConfigValue(config, 'csvCacheEnabled')


def getCacheDir(config):
    """The configured cache directory, created on demand.

    A relative setting is resolved against the current working directory, so the
    default ('backfill') keeps writing to ./backfill.
    """
    cacheDir = os.path.abspath(getConfigValue(config, 'csvCacheDir'))
    os.makedirs(cacheDir, exist_ok=True)
    return cacheDir


def _accountSlug(account):
    """A filesystem-safe stem for this account's cache files.

    Account names are free text, so anything outside a conservative allowlist is
    replaced with '_'. Dropping path separators and dots keeps a name like
    'sub/acct' or '../../etc/cron.d/x' from steering the CSV or the manifest --
    which saveManifest() writes with os.replace() -- outside the cache directory.
    A name that sanitizes away entirely falls back to a fixed stem.
    """
    return re.sub(r'[^A-Za-z0-9_-]', '_', account['name']) or 'account'


def getCacheCsvPath(config, account):
    """The single CSV holding every mirrored point for this account."""
    return os.path.join(getCacheDir(config), '{}.csv'.format(_accountSlug(account)))


def getManifestPath(config, account):
    return os.path.join(getCacheDir(config), '{}.manifest.json'.format(_accountSlug(account)))


def _parseUTC(value):
    if not value:
        return None
    return datetime.datetime.fromisoformat(value)


def _formatUTC(dt):
    if dt is None:
        return None
    return dt.astimezone(datetime.UTC).isoformat()


def writeCsvPoints(csvPath, points):
    """Appends points to a CSV file, writing the header when the file is new."""
    writeHeader = not os.path.exists(csvPath)
    with open(csvPath, 'a', newline='') as f:
        writer = csv.writer(f)
        if writeHeader:
            writer.writerow(CSV_HEADER)
        for pt in points:
            writer.writerow([pt.accountName, pt.deviceName, pt.chanName, pt.usageWatts,
                             pt.timestamp.isoformat(), pt.detailed])


def writeRollupPoints(config, account, points):
    """Mirror freshly collected hourly/daily rollup points into the CSV cache.

    Called from the normal collection loop so the CSV copy stays as current as
    the database. Returns the number of points written.
    """
    if not isCacheEnabled(config) or not points:
        return 0
    csvPath = getCacheCsvPath(config, account)
    writeCsvPoints(csvPath, points)
    logger.debug('Mirrored {} rollup points to CSV cache; csv={}'.format(len(points), csvPath))
    return len(points)


def pointKey(point):
    """Identity of a reading: one value per series per instant per resolution."""
    return (point.deviceName, point.chanName, point.timestamp.isoformat(), str(point.detailed))


def loadMirroredKeys(config, account, sinceUTC):
    """Keys of points already in the CSV at or after sinceUTC.

    Used to keep a backfill from re-appending readings the running daemon has
    already mirrored. sinceUTC should be the coverage end in effect before the
    backfill: rollups only ever write near-now points, and anything older came
    from an earlier backfill, which the gap calculation already excludes. That
    bounds this to the rollup era rather than the whole file.

    Returns an empty set when there is no coverage yet or no file, since neither
    case can contain an overlap.
    """
    csvPath = getCacheCsvPath(config, account)
    if sinceUTC is None or not os.path.exists(csvPath):
        return set()

    since = sinceUTC.isoformat()
    keys = set()
    with open(csvPath, newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 6:
                continue
            if row[4] >= since:  # ISO-8601 UTC sorts lexicographically
                keys.add((row[1], row[2], row[4], row[5]))
    logger.info('Loaded {} already-mirrored CSV keys at or after {}'.format(len(keys), since))
    return keys


def loadManifest(config, account):
    """Return {coveredStartUTC, coveredEndUTC, earliestDataUTC} or None if absent/unreadable."""
    path = getManifestPath(config, account)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            raw = json.load(f)
        return {
            'coveredStartUTC': _parseUTC(raw.get('coveredStartUTC')),
            'coveredEndUTC': _parseUTC(raw.get('coveredEndUTC')),
            'earliestDataUTC': _parseUTC(raw.get('earliestDataUTC')),
        }
    except Exception as e:
        logger.warning('Failed to read cache manifest %s: %s; ignoring', path, e)
        return None


def saveManifest(config, account, coveredStartUTC, coveredEndUTC, earliestDataUTC):
    """Atomically persist the account's cache coverage manifest."""
    path = getManifestPath(config, account)
    data = {
        'coveredStartUTC': _formatUTC(coveredStartUTC),
        'coveredEndUTC': _formatUTC(coveredEndUTC),
        'earliestDataUTC': _formatUTC(earliestDataUTC),
        'updatedUTC': _formatUTC(datetime.datetime.now(datetime.UTC)),
    }
    tmpPath = path + '.tmp'
    with open(tmpPath, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmpPath, path)
    logger.info('Saved history cache manifest; covered=%s..%s; earliestData=%s',
                data['coveredStartUTC'], data['coveredEndUTC'], data['earliestDataUTC'])


def bootstrapManifestFromCsvs(config, account):
    """Seed coverage from an existing CSV when no manifest exists yet.

    Only data rows are present, so the minimum row timestamp is the earliest
    data (and the best lower bound we can claim for coverage) and the maximum is
    the covered end. Returns (coveredStart, coveredEnd, earliestData) or None
    when there is no cached data. This full scan runs at most once, because
    saveManifest() persists the result afterward.

    This is a coarse fallback for a lost manifest: it cannot see gaps, so a
    period the daemon was down for is still reported as covered. Delete the
    manifest and the CSV to force a complete re-probe.
    """
    csvPath = getCacheCsvPath(config, account)
    if not os.path.exists(csvPath):
        return None

    minTs = None
    maxTs = None
    with open(csvPath, newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 5:
                continue
            try:
                ts = datetime.datetime.fromisoformat(row[4])
            except ValueError:
                continue
            if minTs is None or ts < minTs:
                minTs = ts
            if maxTs is None or ts > maxTs:
                maxTs = ts
    if minTs is None:
        return None
    return minTs, maxTs, minTs


def subtractCoverage(reqStartUTC, reqEndUTC, covStartUTC, covEndUTC):
    """Portions of [reqStart, reqEnd] not within [covStart, covEnd].

    Returns up to two (start, stop) intervals: an older 'head' before coverage
    and a newer 'tail' after it. Empty when the request is fully covered.
    """
    if covStartUTC is None or covEndUTC is None:
        return [(reqStartUTC, reqEndUTC)]
    gaps = []
    if reqStartUTC < covStartUTC:
        gaps.append((reqStartUTC, min(reqEndUTC, covStartUTC)))
    if reqEndUTC > covEndUTC:
        gaps.append((max(reqStartUTC, covEndUTC), reqEndUTC))
    return [(s, e) for (s, e) in gaps if s < e]


def mergeCoverage(pieces):
    """Merges (start, stop) intervals into sorted, non-overlapping intervals.

    Abutting pieces (one's stop == the next one's start) are joined, since the
    pieces describe a probed span of the timeline rather than discrete samples.
    Used to distinguish coverage that really is contiguous from pieces separated
    by a hole -- see the manifest handling in collect.collectHistoryUsage.
    """
    merged = []
    for start, stop in sorted(pieces):
        if merged and start <= merged[-1][1]:
            if stop > merged[-1][1]:
                merged[-1] = (merged[-1][0], stop)
        else:
            merged.append((start, stop))
    return merged


def restoreCacheToDatabase(config, account, rangeStartUTC, rangeEndUTC):
    """Stream cached CSV points within [rangeStart, rangeEnd] into the database.

    Routed through the destination layer, so the restore lands in whichever
    databases are configured. Writes are idempotent (same series+timestamp
    overwrites), so any residual duplicates in the file are harmless. Returns
    the number of points restored.
    """
    from vuegraf.collect import Point  # local import avoids a circular import at module load

    csvPath = getCacheCsvPath(config, account)
    if not os.path.exists(csvPath):
        return 0

    buffer = []
    restored = 0

    def flush():
        nonlocal buffer, restored
        if buffer:
            writeDataPoints(config, buffer)  # hand off; start a fresh list so the batch is not mutated
            restored += len(buffer)
            buffer = []

    with open(csvPath, newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 6:
                continue
            try:
                ts = datetime.datetime.fromisoformat(row[4])
            except ValueError:
                continue
            if ts < rangeStartUTC or ts > rangeEndUTC:
                continue
            try:
                watts = float(row[3])
            except ValueError:
                continue
            buffer.append(Point(row[0], row[1], row[2], watts, ts, row[5]))
            if len(buffer) >= RESTORE_CHUNK:
                flush()
    flush()
    return restored
