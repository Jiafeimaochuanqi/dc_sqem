class Node:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

class EdgeNode(Node):
    def __init__(self, data, cavity_id=-1):
        super().__init__(data)
        self.cavity_id = cavity_id
        self.next_vertex_id=-1

    def get_info(self):
        return self.extra_info
class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def append(self, node):
        if not isinstance(node, Node):
            node = Node(node)
            
        if self.is_empty():
            self.head = self.tail = node
        else:
            node.prev = self.tail
            self.tail.next = node
            self.tail = node
        self.size += 1

    def is_empty(self):
        return self.size == 0

    def traverse_forward(self):
        elements = []
        current = self.head
        while current:
            elements.append(str(current.data))
            current = current.next
        return " -> ".join(elements)