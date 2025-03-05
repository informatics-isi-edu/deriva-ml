from datetime import datetime
from deriva.core import urlquote


# -- ==============================================================================================
def get_record_history(server, cid, sname, tname, kvals, kcols=["RID"], snap=None):
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
        l = server.get(url).json()
        if len(l) > 1:
            raise ValueError("got more than one row for %r" % url)
        if len(l) == 0:
            #  sys.stderr.write("ERROR: %s: No record found \n" % (url))
            break
        row = l[0]
        snap2rows[parts["snap"]] = row
        rows_found.append(row)
        rmt = datetime.fromisoformat(row["RMT"])
        # find snap ID prior to row version birth time
        parts["snap"] = urlb32_encode(datetime_epoch_us(rmt) - 1)

    return snap2rows


# -- --------------------------------------------------------------------------------------
def datetime_epoch_us(dt):
    """Return microseconds-since-epoch integer for given timezone-qualified datetime"""
    return int(dt.timestamp()) * 1000000 + dt.microsecond


# -- --------------------------------------------------------------------------------------
# Take the iso format string (same as RMT) and return the version number
#
def iso_to_snap(iso_datetime):
    rmt = datetime.fromisoformat(iso_datetime)
    return urlb32_encode(datetime_epoch_us(rmt))


# -- --------------------------------------------------------------------------------------
def urlb32_encode(i):
    """Encode integer as per ERMrest's base-32 snapshot encoding"""
    if i > 2**63 - 1:
        raise ValueError(i)
    elif i < -(2**63):
        raise ValueError(i)

    # pad 64 bit to 65 bits for 13 5-bit digits
    raw = i << 1
    encoded_rev = []
    for d in range(1, 14):
        if d > 2 and ((d - 1) % 4) == 0:
            encoded_rev.append("-")
        code = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"[raw % 32]
        encoded_rev.append(code)
        raw = raw // 32

    while encoded_rev and encoded_rev[-1] in {"0", "-"}:
        del encoded_rev[-1]

    if not encoded_rev:
        encoded_rev = ["0"]

    encoded = reversed(encoded_rev)

    return "".join(encoded)
