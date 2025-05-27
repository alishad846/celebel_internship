# Node class to represent each element in the linked list
class Node:
    def __init__(self, data):
        self.data = data  # store the value
        self.next = None  # pointer to the next node


# LinkedList class to manage nodes
class LinkedList:
    def __init__(self):
        self.head = None  # start with an empty list

    def add_node(self, data):
        """Add a new node with the given data to the end of the list."""
        new_node = Node(data)
        if not self.head:
            # If the list is empty, the new node becomes the head
            self.head = new_node
            return
        # Otherwise, traverse to the end and append the new node
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def print_list(self):
        """Display the entire list in a readable format."""
        if not self.head:
            print("The list is empty.")
            return
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")  # Marks the end of the list

    def delete_nth_node(self, n):
        """Delete the nth node in the list (1-based index)."""
        try:
            if not self.head:
                raise IndexError("Cannot delete from an empty list.")

            if n <= 0:
                raise ValueError("Position must be a positive integer.")

            # Deleting the first node (special case)
            if n == 1:
                self.head = self.head.next
                return

            current = self.head
            count = 1

            # Traverse to the (n-1)th node
            while current and count < n - 1:
                current = current.next
                count += 1

            # If next node is None, index is out of range
            if not current or not current.next:
                raise IndexError("Index out of range.")

            # Skip the nth node
            current.next = current.next.next

        except (IndexError, ValueError) as error:
            print(f"Error: {error}")


# Testing the LinkedList implementation
if __name__ == "__main__":
    # Create a new linked list
    my_list = LinkedList()

    # Add some elements
    my_list.add_node(10)
    my_list.add_node(20)
    my_list.add_node(30)
    my_list.add_node(40)

    print("Initial list:")
    my_list.print_list()

    # Delete the second node
    print("\nDeleting 2nd node:")
    my_list.delete_nth_node(2)
    my_list.print_list()

    # Delete the first node
    print("\nDeleting 1st node:")
    my_list.delete_nth_node(1)
    my_list.print_list()

    # Try to delete an out-of-range node
    print("\nTrying to delete 5th node (should fail):")
    my_list.delete_nth_node(5)

    # Delete remaining nodes
    my_list.delete_nth_node(1)
    my_list.delete_nth_node(1)

    # Try deleting from an empty list
    print("\nTrying to delete from empty list (should fail):")
    my_list.delete_nth_node(1)
