import sys,os

def run():
#	generateString()
	i = 5
	p = 5.001
	q = 5.0000000000000323232
	print('i',sys.getsizeof(['Hi']))
	print('i',sys.getsizeof(('Hi')))
	print('i',sys.getsizeof({'Hi':1}))
	print('i',sys.getsizeof(i))
	print('p',sys.getsizeof(p))
	print('q',sys.getsizeof(q))

	
def generateString():
	for each in range(212):
		str_1 = "`"+str(each)+"`!=0 or "
		print(str_1)
