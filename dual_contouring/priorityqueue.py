import heapq

class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.entry_map = {}
        self.counter = 0

    def push(self, item, priority):
        if item in self.entry_map:
            self._remove_item(item)
        entry = [priority, self.counter, item]
        heapq.heappush(self.heap, entry)
        self.entry_map[item] = entry
        self.counter += 1

    def _remove_item(self, item):
        entry = self.entry_map.pop(item)
        entry[-1] = None

    def pop(self):
        while self.heap:
            priority, _, item = heapq.heappop(self.heap)
            if item is not None:
                del self.entry_map[item]
                return item,priority
        raise IndexError("queue is empty")

    def update_priority(self, item, new_priority):
        if self.contains(item):
            self.push(item, new_priority)

    def is_empty(self):
        return len(self.entry_map) == 0
    def __bool__(self):
        return len(self.entry_map) > 0


    def __len__(self):
        return len(self.entry_map)
    def traverse_via_map(self):
        return list(self.entry_map.keys())
    def contains(self, key):
        return key in self.entry_map
    def items_sorted(self):
        valid_entries = [entry for entry in self.heap if entry[-1] is not None]
        valid_entries.sort(key=lambda entry: (entry[0], entry[1]))
        return [(entry[2], entry[0]) for entry in valid_entries]


import heapq
import threading
'''
Use a heap to maintain the pop order, a hash table to maintain the current valid state, and a counter to ensure stable order under the same priority.
'''
class MulPriorityQueue:
    def __init__(self, is_thread_safe=False):
        self.heap = []
        self.entry_map = {}  # item -> [priority_tuple, counter, item],list(self.entry_map.keys()) get keys according insert order
        self.counter = 0
        self.lock = threading.Lock() if is_thread_safe else None
    
    def _synchronized(fn):
        def wrapper(self, *args, **kwargs):
            if self.lock:
                with self.lock:
                    return fn(self, *args, **kwargs)
            return fn(self, *args, **kwargs)
        return wrapper

    @_synchronized
    def push(self, item, priority_tuple):
        if not isinstance(priority_tuple, tuple):
            raise TypeError("priority must is tuple")
        if item in self.entry_map:
            old_entry = self.entry_map[item]
            if old_entry[0] == priority_tuple:
                return
        
        if item in self.entry_map:
            self._remove_item(item)
   
        entry = [priority_tuple, self.counter, item]
        heapq.heappush(self.heap, entry)
        self.entry_map[item] = entry
        self.counter += 1
    def update_priority(self, item, new_priority_tuple):
        if self.contains(item):
            self.push(item, new_priority_tuple)


    def _remove_item(self, item):
        entry = self.entry_map.pop(item)
        entry[-1] = None

    @_synchronized
    def pop(self):
        while self.heap:
            entry = heapq.heappop(self.heap)
            _, _, item = entry
            if item is not None:
                if item in self.entry_map:
                    del self.entry_map[item]
                return item, entry[0]
        raise IndexError("queue is empty")

    @_synchronized
    def contains(self, item):
        return item in self.entry_map

    def __len__(self):
        return len(self.entry_map)
    def traverse_via_map(self):
        return list(self.entry_map.keys())
    def is_empty(self):
        return len(self.entry_map) == 0
    @_synchronized
    def items_sorted(self):
        valid_entries = [entry for entry in self.heap if entry[-1] is not None]
        valid_entries.sort(key=lambda entry: (entry[0], entry[1]))
        return [(entry[2], entry[0]) for entry in valid_entries]