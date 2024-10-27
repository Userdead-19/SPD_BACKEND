import turtle

class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.value = key
		
class MinMaxHeap(object):
	def __init__(self, reserve=0):
		self.a = [None] * reserve
		self.size = 0

	def __len__(self):
		return self.size

	def insert(self, key):
		"""
		Insert key into heap. Complexity: O(log(n))
		"""
		if len(self.a) < self.size + 1:
			self.a.append(key)
		insert(self.a, key, self.size)
		self.size += 1

	def peekmin(self):
		"""
		Get minimum element. Complexity: O(1)
		"""
		return peekmin(self.a, self.size)

	def peekmax(self):
		"""
		Get maximum element. Complexity: O(1)
		"""
		return peekmax(self.a, self.size)

	def popmin(self):
		"""
		Remove and return minimum element. Complexity: O(log(n))
		"""
		m, self.size = removemin(self.a, self.size)
		return m

	def popmax(self):
		"""
		Remove and return maximum element. Complexity: O(log(n))
		"""
		m, self.size = removemax(self.a, self.size)
		return m

	def visualize_tree_with_turtle(self):
		if not self.a:
			print("The heap is empty. No visualization will be shown.")
			return 
		root = self._build_tree(0)  # Build the tree from the heap array
		self._clear_screen()  # Clear the previous drawing
		self._visualize_tree(root)
	def _build_tree(self, index):
		if index >= self.size:
			return None
		node = Node(self.a[index])
		node.left = self._build_tree(2 * index + 1)
		node.right = self._build_tree(2 * index + 2)
		return node
	def _visualize_tree(self, root):
		pen = turtle.Turtle()
		pen.speed(0)
		pen.hideturtle()
		draw_tree(root, 0, 250, 90, 0, 200, pen)
		turtle_screen.update()
		
	def _clear_screen(self):
		turtle_screen.clear()
def draw_tree(node, x, y, angle, depth, length, pen):
    if node is None:
        return

    # Move to position of the current node and draw the circle
    pen.penup()
    pen.goto(x, y)
    pen.pendown()

    # Set color for the node: sky blue fill with black outline
    pen.fillcolor("sky blue")
    pen.pencolor("black")
    
    pen.begin_fill()
    pen.circle(20)  # Draw circle for the node
    pen.end_fill()

    # Draw node value in black (adjust the y position)
    pen.penup()
    pen.goto(x, y + 5)  # Adjust y position to center the value better
    pen.pendown()
    pen.pencolor("black")
    pen.write(node.value, align="center", font=("Arial", 12, "bold"))  # Write node value
    
    # Calculate the position for the left and right children
    if node.left:
        new_x = x - length * (1.5 ** -depth)
        new_y = y - 100
        pen.penup()
        pen.goto(x - 20, y - 20)
        pen.pendown()
        pen.goto(new_x, new_y + 20)
        draw_tree(node.left, new_x, new_y, angle + 30, depth + 1, length, pen)

    if node.right:
        new_x = x + length * (1.5 ** -depth)
        new_y = y - 100
        pen.penup()
        pen.goto(x + 20, y - 20)
        pen.pendown()
        pen.goto(new_x, new_y + 20)
        draw_tree(node.right, new_x, new_y, angle - 30, depth + 1, length, pen)

# Create a Turtle screen and manage its state
turtle_screen = turtle.Screen()
turtle_screen.title("Min-Max Heap Visualization")
turtle_screen.tracer(0)  # Disable automatic screen updates for manual control


def level(i):
	return (i+1).bit_length() - 1


def trickledown(array, i, size):
	if level(i) % 2 == 0:  # min level
		trickledownmin(array, i, size)
	else:
		trickledownmax(array, i, size)


def trickledownmin(array, i, size):
	if size > i * 2 + 1:  # i has children
		m = i * 2 + 1
		if i * 2 + 2 < size and array[i*2+2] < array[m]:
			m = i*2+2
		child = True
		for j in range(i*4+3, min(i*4+7, size)):
			if array[j] < array[m]:
				m = j
				child = False

		if child:
			if array[m] < array[i]:
				array[i], array[m] = array[m], array[i]
		else:
			if array[m] < array[i]:
				if array[m] < array[i]:
					array[m], array[i] = array[i], array[m]
				if array[m] > array[(m-1) // 2]:
					array[m], array[(m-1)//2] = array[(m-1)//2], array[m]
				trickledownmin(array, m, size)


def trickledownmax(array, i, size):
	if size > i * 2 + 1:  # i has children
		m = i * 2 + 1
		if i * 2 + 2 < size and array[i*2+2] > array[m]:
			m = i*2+2
		child = True
		for j in range(i*4+3, min(i*4+7, size)):
			if array[j] > array[m]:
				m = j
				child = False

		if child:
			if array[m] > array[i]:
				array[i], array[m] = array[m], array[i]
		else:
			if array[m] > array[i]:
				if array[m] > array[i]:
					array[m], array[i] = array[i], array[m]
				if array[m] < array[(m-1) // 2]:
					array[m], array[(m-1)//2] = array[(m-1)//2], array[m]
				trickledownmax(array, m, size)


def bubbleup(array, i):
	if level(i) % 2 == 0:  # min level
		if i > 0 and array[i] > array[(i-1) // 2]:
			array[i], array[(i-1) // 2] = array[(i-1)//2], array[i]
			bubbleupmax(array, (i-1)//2)
		else:
			bubbleupmin(array, i)
	else:  # max level
		if i > 0 and array[i] < array[(i-1) // 2]:
			array[i], array[(i-1) // 2] = array[(i-1) // 2], array[i]
			bubbleupmin(array, (i-1)//2)
		else:
			bubbleupmax(array, i)


def bubbleupmin(array, i):
	while i > 2:
		if array[i] < array[(i-3) // 4]:
			array[i], array[(i-3) // 4] = array[(i-3) // 4], array[i]
			i = (i-3) // 4
		else:
			return


def bubbleupmax(array, i):
	while i > 2:
		if array[i] > array[(i-3) // 4]:
			array[i], array[(i-3) // 4] = array[(i-3) // 4], array[i]
			i = (i-3) // 4
		else:
			return


def peekmin(array, size):
	assert size > 0
	return array[0]


def peekmax(array, size):
	assert size > 0
	if size == 1:
		return array[0]
	elif size == 2:
		return array[1]
	else:
		return max(array[1], array[2])


def removemin(array, size):
	assert size > 0
	elem = array[0]
	array[0] = array[size-1]
	# array = array[:-1]
	trickledown(array, 0, size - 1)
	return elem, size-1


def removemax(array, size):
	assert size > 0
	if size == 1:
		return array[0], size - 1
	elif size == 2:
		return array[1], size - 1
	else:
		i = 1 if array[1] > array[2] else 2
		elem = array[i]
		array[i] = array[size-1]
		# array = array[:-1]
		trickledown(array, i, size - 1)
		return elem, size-1


def insert(array, k, size):
	array[size] = k
	bubbleup(array, size)


def minmaxheapproperty(array, size):
	for i, k in enumerate(array[:size]):
		if level(i) % 2 == 0:  # min level
			# check children to be larger
			for j in range(2 * i + 1, min(2 * i + 3, size)):
				if array[j] < k:
					print(array, j, i, array[j], array[i], level(i))
					return False
			# check grand children to be larger
			for j in range(4 * i + 3, min(4 * i + 7, size)):
				if array[j] < k:
					print(array, j, i, array[j], array[i], level(i))
					return False
		else:
			# check children to be smaller
			for j in range(2 * i + 1, min(2 * i + 3, size)):
				if array[j] > k:
					print(array, j, i, array[j], array[i], level(i))
					return False
			# check grand children to be smaller
			for j in range(4 * i + 3, min(4 * i + 7, size)):
				if array[j] > k:
					print(array, j, i, array[j], array[i], level(i))
					return False

	return True

def main():
    heap = MinMaxHeap()

    while True:
        print("\nMenu:")
        print("1. Insert an element")
        print("2. Delete minimum element")
        print("3. Delete maximum element")
        print("4. Display heap")
        print("5. Get minimum element")
        print("6. Get maximum element")
        print("7. Exit")
        
        heap.visualize_tree_with_turtle()  # Visualize tree before every menu operation
        choice = input("Enter your choice (1-7): ")

        if choice == '1':
            element = int(input("Enter the element to insert: "))
            heap.insert(element)
            print(f"Inserted {element}.")
        
        elif choice == '2':
            min_elem = heap.popmin()  # Corrected function call
            if min_elem is not None:
                print(f"Deleted minimum element: {min_elem}")
            else:
                print("Heap is empty, cannot delete minimum element.")

        elif choice == '3':
            max_elem = heap.popmax()  # Corrected function call
            if max_elem is not None:
                print(f"Deleted maximum element: {max_elem}")
            else:
                print("Heap is empty or has only one element, cannot delete maximum element.")

        elif choice == '4':
            print(heap.a[:heap.size])  # Displaying the heap content (you might want to implement a more formatted display)

        elif choice == '5':
            min_elem = heap.peekmin()  # Corrected function call
            if min_elem is not None:
                print(f"Minimum element: {min_elem}")
            else:
                print("Heap is empty.")

        elif choice == '6':
            max_elem = heap.peekmax()  # Corrected function call
            if max_elem is not None:
                print(f"Maximum element: {max_elem}")
            else:
                print("Heap is empty or has only one element.")

        elif choice == '7':
            print("Exiting...")
            turtle_screen.bye()  # Close the turtle graphics window
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()