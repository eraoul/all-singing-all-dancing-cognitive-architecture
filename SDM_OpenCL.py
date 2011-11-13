import pyopencl as cl
import pyopencl.array 
import numpy
import numpy.linalg as la
import time

DIMENSIONS = 256
HARD_LOCATIONS = 2**20
INT_SIZE = 256 # length of memory buffer that each kernel receives
NUM_INTS = DIMENSIONS/INT_SIZE # why?  Because N=256 / 32 bits (usando uint32)
SIZE = HARD_LOCATIONS * NUM_INTS  # total size of hard loation memory buffer in 
ACCESS_RADIUS_THRESHOLD = 104

print "Size=", SIZE 
print "Dimensions=", DIMENSIONS
print 'Num of INTs=', NUM_INTS
print 'INT size=', INT_SIZE

numpy.random.seed(seed=1234567890)

# comments

if INT_SIZE == 32: 
  TYPE = numpy.uint32 # cl.array.vec.uint8 #
  memory_addresses = numpy.random.random_integers(0,2**INT_SIZE,size=SIZE).astype(TYPE)
  bitstring = numpy.random.random_integers(0,2**INT_SIZE,size=NUM_INTS).astype(TYPE)
elif INT_SIZE == 64:
  TYPE = numpy.uint64
  memory_addresses = numpy.random.random_integers(0,2**INT_SIZE,size=SIZE).astype(TYPE)
  bitstring = numpy.random.random_integers(0,2**INT_SIZE,size=NUM_INTS).astype(TYPE)
elif INT_SIZE == 256:
  TYPE = pyopencl.array.vec.uint8
  memory_addresses = numpy.random.random_integers(0,2**32,size=(2**22)*8).astype(numpy.uint32) ##WTF???
  bitstring = numpy.random.random_integers(0,2**32,size=NUM_INTS * 8).astype(numpy.uint32)  ## WTF???
 
semaphor = numpy.zeros(10).astype(numpy.uint32) 

active_hard_locations = numpy.zeros(1025).astype(numpy.uint32) #after semaphores, down to 1025
hamming_distances = numpy.zeros(1025).astype(numpy.uint32) #after semaphores, down to 1025
hamming_distances[0] = ACCESS_RADIUS_THRESHOLD

print 'ACCESS_RADIUS_THRESHOLD=', hamming_distances[0]

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mem_flags = cl.mem_flags
memory_addresses_buffer = cl.Buffer(ctx, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=memory_addresses)
#hamming_distances_buffer = cl.Buffer(ctx, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=hamming_distances)
semaphor_buffer = cl.Buffer(ctx, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=semaphor)
bitstring_buf = cl.Buffer(ctx, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=bitstring)

active_hard_locations_buffer = cl.Buffer(ctx, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, active_hard_locations.nbytes, hostbuf=active_hard_locations)

hamming_distances_buffer = cl.Buffer(ctx, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hamming_distances.nbytes, hostbuf=hamming_distances)

start = time.time()

prg = cl.Program(ctx, """
    //For N=256, use ulong4, if N=1024, use ulong16.
    const uint MaxActiveHLs =1025;

    void GetSemaphor(__global uint *semaphor) 
    {
        int occupied = atom_xchg(semaphor, 1);
        while(occupied > 0)  occupied = atom_xchg(semaphor, 1);
    }

    void ReleaseSemaphor(__global uint *semaphor)
    {
      int prevVal = atom_xchg(semaphor, 0);
    }

    int pop_count_uint32(uint i)
    {
      i = i - ((i >> 1) & 0x55555555);
      i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
      return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
    }

    __kernel void get_active_hard_locations_256bits_in_uint8(__global uint8 *hard_location_addresses,
    __global uint8 *bitstring, __global uint *activeHLs, __global uint *hamming_distances, __global uint *semaphor)
    {
        uint8 Xor;
        uint8 local_bitstring = bitstring[0];
        uint hamming = 0;
        uint gid = get_global_id(0);
        uint8 local_hard_location_address = hard_location_addresses[gid];
      
        Xor = local_hard_location_address ^ bitstring[0]; 

        hamming += pop_count_uint32 (Xor.s0);
        hamming += pop_count_uint32 (Xor.s1);
        hamming += pop_count_uint32 (Xor.s2);
        hamming += pop_count_uint32 (Xor.s3);
        hamming += pop_count_uint32 (Xor.s4);
        hamming += pop_count_uint32 (Xor.s5);
        hamming += pop_count_uint32 (Xor.s6);
        hamming += pop_count_uint32 (Xor.s7);

        if (activeHLs[0]<(MaxActiveHLs-1) && hamming<hamming_distances[0] )  //104 is the one: 128-24: mu-3sigma. Com o seed = 1234567890, gera 1153 Active Hard Locations
        {
        //      GetSemaphor(&semaphor[0]);
              //{
                  activeHLs[0]=activeHLs[0]+1;
                  activeHLs[activeHLs[0]]=gid;
                  hamming_distances[activeHLs[0]]=hamming;
              //}
        //  ReleaseSemaphor(&semaphor[0]);
        }
    }""").build()

#print 'Time to build GPU code:', (time.time()-start)

print INT_SIZE,'bits,   ', DIMENSIONS, ' dimensions,   2^20 Hard Locations'

start = time.time()
for x in range(10):
    prg.get_active_hard_locations_256bits_in_uint8(queue, (HARD_LOCATIONS,), None, memory_addresses_buffer, bitstring_buf, active_hard_locations_buffer, hamming_distances_buffer, semaphor_buffer).wait()  
print 'Time to compute some Hamming distances 10 times:', (time.time()-start)

#cl.enqueue_read_buffer(queue, dist_buf, dist).wait()
err = cl.enqueue_read_buffer(queue, active_hard_locations_buffer, active_hard_locations).wait()
err = cl.enqueue_read_buffer(queue, hamming_distances_buffer, hamming_distances).wait()

print active_hard_locations
print hamming_distances