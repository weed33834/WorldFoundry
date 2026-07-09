"""Module for base_models -> three_dimensions -> depth -> moge -> utils -> webfile.py functionality."""

import requests  
from typing import *  
  
__all__ = ["WebFile"]


class WebFile:  
    """Web file implementation."""
    def __init__(self, url: str, session: Optional[requests.Session] = None, headers: Optional[Dict[str, str]] = None, size: Optional[int] = None):  
        """Init.

        Args:
            url: The url.
            session: The session.
            headers: The headers.
            size: The size.
        """
        self.url = url  
        self.session = session or requests.Session()  
        self.session.headers.update(headers or {})
        self._offset = 0  
        self.size = size if size is not None else self._fetch_size()
  
    def _fetch_size(self):  
        """Helper function to fetch size."""
        with self.session.get(self.url, stream=True) as response:  
            response.raise_for_status()  
            content_length = response.headers.get("Content-Length")  
            if content_length is None:  
                raise ValueError("Missing Content-Length in header")  
            return int(content_length) 

    def _fetch_data(self, offset: int, n: int) -> bytes:
        """Helper function to fetch data.

        Args:
            offset: The offset.
            n: The n.

        Returns:
            The return value.
        """
        headers = {"Range": f"bytes={offset}-{min(offset + n - 1, self.size)}"}
        response = self.session.get(self.url, headers=headers)
        response.raise_for_status()
        return response.content
  
    def seekable(self) -> bool:  
        """Seekable.

        Returns:
            The return value.
        """
        return True  
  
    def tell(self) -> int:  
        """Tell.

        Returns:
            The return value.
        """
        return self._offset  
  
    def available(self) -> int:  
        """Available.

        Returns:
            The return value.
        """
        return self.size - self._offset  
  
    def seek(self, offset: int, whence: int = 0) -> None:  
        """Seek.

        Args:
            offset: The offset.
            whence: The whence.

        Returns:
            The return value.
        """
        if whence == 0:  
            new_offset = offset  
        elif whence == 1:  
            new_offset = self._offset + offset  
        elif whence == 2:  
            new_offset = self.size + offset  
        else:  
            raise ValueError("Invalid value for whence")  

        self._offset = max(0, min(new_offset, self.size))  
  
    def read(self, n: Optional[int] = None) -> bytes:  
        """Read.

        Args:
            n: The n.

        Returns:
            The return value.
        """
        if n is None or n < 0:  
            n = self.available()  
        else:  
            n = min(n, self.available())  

        if n == 0:  
            return b''  

        data = self._fetch_data(self._offset, n)
        self._offset += len(data)  
  
        return data  

    def close(self) -> None:  
        """Close.

        Returns:
            The return value.
        """
        pass

    def __enter__(self):  
        """Enter."""
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Exit.

        Args:
            exc_type: The exc type.
            exc_value: The exc value.
            traceback: The traceback.
        """
        pass
  
    
