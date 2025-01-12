"""Memory store implementation."""
from typing import Any, Optional, Tuple
from langgraph.store.base import BaseStore

class MemoryStore(BaseStore):
    """Simple in-memory store implementation."""
    
    def __init__(self):
        """Initialize the store."""
        self._store = {}

    async def aget(
        self,
        namespace: Tuple[str, str],
        key: str,
    ) -> Optional[Any]:
        """Get a value from the store."""
        return self._store.get((namespace, key))

    async def aput(
        self,
        namespace: Tuple[str, str],
        key: str,
        value: Any,
    ) -> None:
        """Put a value in the store."""
        self._store[(namespace, key)] = value

    async def adelete(
        self,
        namespace: Tuple[str, str],
        key: str,
    ) -> None:
        """Delete a value from the store."""
        self._store.pop((namespace, key), None)

    async def alist(
        self,
        namespace: Tuple[str, str],
    ) -> list[str]:
        """List all keys in a namespace."""
        return [
            key for (ns, key) in self._store.keys()
            if ns == namespace
        ]

    async def asearch(
        self,
        namespace: Tuple[str, str],
        query: Optional[str] = None,
        limit: int = 10,
    ) -> list[Any]:
        """Search for values in the store."""
        items = [
            value for (ns, _), value in self._store.items()
            if ns == namespace
        ]
        return items[:limit]