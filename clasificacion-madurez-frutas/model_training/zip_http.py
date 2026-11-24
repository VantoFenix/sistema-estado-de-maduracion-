from io import SEEK_CUR, SEEK_END, SEEK_SET
import urllib.request


class RangeReader:
    def __init__(self, url):
        self.url = url
        self.offset = 0
        request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=20) as response:
            range_header = response.headers["Content-Range"]
            self.size = int(range_header.rsplit("/", 1)[1])
            self.url = response.geturl()

    def seekable(self):
        return True

    def tell(self):
        return self.offset

    def seek(self, offset, whence=SEEK_SET):
        if whence == SEEK_END:
            offset += self.size
        elif whence == SEEK_CUR:
            offset += self.offset
        if offset < 0:
            raise OSError("invalid seek")
        self.offset = offset
        return offset

    def read(self, size=-1):
        remaining = self.size - self.offset
        amount = remaining if size < 0 else min(size, remaining)
        if amount <= 0:
            return b""
        data = self.read_range(self.offset, amount)
        self.offset += amount
        return data

    def read_range(self, start, amount):
        end = start + amount - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=25) as response:
            expected = f"bytes {start}-{end}/{self.size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise OSError("server did not return the requested range")
            data = response.read(amount + 1)
        if len(data) != amount:
            raise OSError("incomplete range")
        return data
