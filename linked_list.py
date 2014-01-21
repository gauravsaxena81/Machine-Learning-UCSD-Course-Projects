class node:
	def __init__(self,data):
		self.data=data
		self.next=None

	def AddNode( self, data ):
  		new_node = Node( data )

  		if self.head == None:
    	self.head = new_node

  		if self.tail != None:
    	self.tail.next = new_node

  		self.tail = new_node