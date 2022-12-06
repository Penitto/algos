class Node:

    def __init__(self, next: None, prev: None, value: int):
        self.next = next
        self.prev = prev
        self.value = value


class SinglyLinkedList:

    def __init__(self, head: Node) -> None:
        self.head = head

    def printList(self):

        tmp = self.head

        while tmp:
            print(tmp.value)
            tmp = tmp.next

    def appendNode(self, node: Node):

        tmp = self.head

        while tmp.next:
            tmp = tmp.next

        tmp.next = node

    def popNode(self):

        tmp = self.head

        while tmp.next.next:
            tmp = tmp.next

        tmp.next = None

    def insertNode(self, node: Node, position: int):

        tmp = self.head

        while position - 1:
            tmp = tmp.next
            position -= 1

        node.next = tmp.next
        tmp.next = node


class DoublyLinkedList:

    def __init__(self, head: Node, tail: Node):
        self.head = head
        self.tail = tail

    def printListL2R(self):

        tmp = self.head

        while tmp:
            print(tmp.value)
            tmp = tmp.next

    def printListR2L(self):

        tmp = self.tail

        while tmp:
            print(tmp.value)
            tmp = tmp.prev

    def appendRight(self, node: Node):

        tmp = self.tail

        node.prev = tmp
        tmp.next = node

        self.tail = node

    def appendLeft(self, node: Node):

        tmp = self.head

        node.next = tmp
        tmp.prev = node

        self.head = node

    def popRight(self):

        tmp = self.tail

        tmp.prev.next = None
        self.tail = tmp.prev

    def popLeft(self):

        tmp = self.head

        tmp.next.prev = None
        self.head = tmp.next

    def insertNode(self, node: Node, position: int):

        tmp = self.root

        while position - 1:
            tmp = tmp.next
            position -= 1

        node.next = tmp.next
        node.prev = tmp
        tmp.next.next.prev = node
        tmp.next = node
