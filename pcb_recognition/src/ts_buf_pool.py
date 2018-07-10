"""
@Author Matthew Noseworthy
@Email  mjn327@mun.ca
@Brief Quick and dirty thread-safe memory management

@Description
Usage:
    ... App init ...
    with BufferPoolResource(Number_of_threads_or_buffers, "batched"|"pooled", callback_pointer ) as buffer_pool:
        ... kick off threads, passing buffer_pool along to each ...

    ... in child thread ...
    buf = buffer_pool.make()
    ... do work ...
    buffer_pool.writeout(data, buf)
    
From there either,
    If callback passed to init & is batched:
        ... callback executes once one batch of buffers has released ...
        process the data given
    else if batched:
        ... Stack requesting data ...
        if buffer_pool.batch_complete():
            buffer_pool.getBatch()
    else:
        ... stack accessing data ...
        buffer_pool.get(index)
"""
from collections import Counter
from multiprocessing import Lock, Queue, Value

class BufferPoolResource:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    def __enter__(self):
        class BufferPool():
            """
                Handles memory output from parallel filtering of input images
                in @PCB_Dataset.parseImages
            """
            def __init__(self, length, config, callback=None):
                """
                    init
                """
                # Parse input params

                if(config == "batched"):
                    self.batched, self.pooled = (True, False)
                elif(config == "pooled"):
                    self.batched, self.pooled = (False, True)
                else:
                    raise ValueError("Config must currently either be 'batched' or 'pooled'", config, length, callback)
                
                self.callback = callback

                #buffer_struct = {
                #    "index" : 
                #}

                # list of buffer_struct's
                self.bufferPool  = Queue()
                self.outputQueue = Queue()
                # Initialize queue
                for i in range(length):

                    newObj = {
                        "leased" : False,
                        "data" : None,
                        "Rx_count" : 0,
                        "written" : False,
                        "index" : None
                    }
                    self.bufferPool.put(newObj)
                #print(len(self.buffer), length )

                # Internals
                self.length = length
                self.written = Value("i", 0)
                self.leased = Value("i", 0)
                self.running = Value("i", 1)



            #
            #   @brief Callback to write data
            #
            def __make_callback(self, index, lock, buf, data):
                lock.acquire()
                try:
                    buf["data"] = data
                    buf["written"] = True 
                    self.written.value += 1
                    self.outputQueue.put(buf, False)
                    #print("Modified buffer {} : {}".format(index,self.buffer[index]))

                    print("Data written to buffer {0}".format(index))

                    if self.batched:
                        # If batched, check if the batch is complete
                        if self.batch_complete():
                            self.join()
                    else:
                        buf["leased"] = 0
                        self.bufferPool.put(buf)
                finally:
                    lock.release()

                
            
            def make(self, lock, index):
                """
                    @brief when called returns a handle that can be used to write to
                    the buffer, and a callback to release the data

                    @param seed is any unique data that identifies the caller
                    
                    @ret (function pointer, buffer_index) - Hold this value in the thread,
                    when processing has finished and you wish to write-out to the buffer
                    pass it to self.__writeout with your data as the second parameter:

                        Ex:
                            bufPool = _image_buffer(50)
                            myBuffer = __make()
                            ... processing ...
                            data = 0xff
                            ... processing ...
                            bufPool.__writeout(myBuffer, data)
                            
                """


                # Lock while allocating an index
                lock.acquire()
                try:
                    print("Process claimed lock")
                    buf = self.bufferPool.get()

                    # Found index to lease, increment lease count
                    self.leased.value  += 1
                    print("Leasing index ", index)
                    buf["index"] = index
                    buf["leased"] = True
                    return ( self.__make_callback, index, lock, buf )
                finally:
                    lock.release()
            

            def writeout(self, data, *eventData):
                """
                    @Brief  Unpacks the tuple of data and executes __make_callback 
                    with the argumetns
                    @Param eventData:
                        (@ret from self.__make() , data_to_write)
                """
                callback, index, lock, buf = eventData[0]
                callback(index, lock, buf, data)
            


            #
            #   @Brief If buffer is not leased, and there is data available, returns the data, False otherwise. 
            #   Reading the buffer releases the memory and if pooled, it will be overwritten.
            #
            def get(self, index):
                """
                    Retreives data
                """
                return False
            
            def join(self, event=None):
                """
                    @Brief If batched, join will be called to handle running the
                    callback with the required arguments, then all memory will be
                    released and this object destroyed
                """
                if event == "sysexit":
                    if self.batched and not self.batch_complete():
                        self.running = False
                        raise ValueError("Object deleted before batch completed, data lost\n Event: {}".format(event) )
                    else:
                        self.running = False
                else:
                    if self.callback != None:
                        # Aggregate data
                        tmp = []
                        for i in range(self.length) :
                            print(i)
                            with self.outputQueue.get(False) as buf:
                                if isinstance(buf, dict):
                                    print("Adding data from index {} to output".format(buf["index"]))
                                    tmp[ buf["index"] ] = buf["data"]
                                # Release buffer if pooled
                                if self.pooled:
                                    buf["leased"] = False
                                    self.bufferPool.put(buf)
                        # return data
                        self.callback(tmp)
                        # Remove callback
                        self.callback = None
            

            def batch_complete(self):
                """
                    @Brief Returns true if the batch is complete, false otherwise
                """
                # Try quick ways of checking first
                print(self.written.value, self.length)
                if self.written.value == self.length:
                    print("Returned true")
                    return True
                else:
                    return False


        ### __enter__()
        #print(self.args)
        #print(self.kwargs)
        self.buffer_pool = BufferPool(*self.args, **self.kwargs)
        return self.buffer_pool

    def __exit__(self, exc_type, exc_value, traceback):
        # Attempt to cleanup
        try:
            self.buffer_pool.join("sysexit")
            while(self.buffer_pool.running):
                pass
            self.buffer_pool = None
        except Exception, err:
            pass
        
