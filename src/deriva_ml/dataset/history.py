import base64
import struct
from datetime import datetime

from dateutil.parser import isoparse
from deriva.core import urlquote


# -- ==============================================================================================
def get_record_history(server, cid, sname, tname, kvals, kcols=["RID"], snap=None):
    """Get the history of a record from the catalog.
    
    Args:
        server: The server instance.
        cid: The catalog ID.
        sname: The schema name.
        tname: The table name.
        kvals: The key values to look up.
        kcols: The key columns. Defaults to ["RID"].
        snap: Optional snapshot ID.
        
    Returns:
        The history data for the record.
        
    Raises:
        ValueError: If more than one row is returned.
    """
    parts = {
        "cid": urlquote(cid),
        "sname": urlquote(sname),
        "tname": urlquote(tname),
        "filter": ",".join(
            [
                "%s=%s" % (urlquote(kcol), urlquote(kval))
                for kcol, kval in zip(kcols, kvals)
            ]
        ),
    }

    if snap is None:
        # determinate starting (latest) snapshot
        r = server.get("/ermrest/catalog/%(cid)s" % parts)
        snap = r.json()["snaptime"]
    parts["snap"] = snap

    path = "/ermrest/catalog/%(cid)s@%(snap)s/entity/%(sname)s:%(tname)s/%(filter)s"

    rows_found = []
    snap2rows = {}
    while True:
        url = path % parts
        # sys.stderr.write("%s\n" % url)
        response_data = server.get(url).json()
        if len(response_data) > 1:
            raise ValueError("got more than one row for %r" % url)
        if len(response_data) == 0:
            #  sys.stderr.write("ERROR: %s: No record found \n" % (url))
            break
        row = response_data[0]
        snap2rows[parts["snap"]] = row
        rows_found.append(row)
        rmt = datetime.fromisoformat(row["RMT"])
        # find snap ID prior to row version birth time
        parts["snap"] = urlb32_encode(datetime_epoch_us(rmt) - 1)

    return snap2rows


# -- --------------------------------------------------------------------------------------
def datetime_epoch_us(dt):
    """Convert datetime to epoch microseconds.
    
    Args:
        dt: The datetime object to convert.
        
    Returns:
        The epoch time in microseconds.
    """
    return int(dt.timestamp() * 1000000)


# -- --------------------------------------------------------------------------------------
# Take the iso format string (same as RMT) and return the version number
#


def iso_to_snap(iso_datetime):
    """Convert ISO datetime string to snapshot format.
    
    Args:
        iso_datetime: The ISO datetime string.
        
    Returns:
        The snapshot timestamp.
    """
    return datetime_epoch_us(isoparse(iso_datetime))


# -- --------------------------------------------------------------------------------------
def urlb32_encode(i):
    """Encode an integer to URL-safe base32.
    
    Args:
        i: The integer to encode.
        
    Returns:
        The URL-safe base32 encoded string.
    """
    return base64.urlsafe_b64encode(struct.pack(">Q", i)).decode("ascii").rstrip("=")
