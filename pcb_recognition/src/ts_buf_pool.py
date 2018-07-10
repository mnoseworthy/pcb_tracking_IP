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
                self.buffer  = range(length) 
                # pool of index's
                self.indexPool = [] 
                for i in range(length):

                    self.buffer[i] = {
                        "leased" : False,
                        "data" : None,
                        "Rx_count" : 0,
                        "written" : False
                    }
                    self.indexPool.append(i)
                



                # Internals
                self.length = length
                self.leased = 0 # Number of successful calls to @__make
                self.running = True


            #
            #   @brief Callback to write data
            #
            def __make_callback(self, index, data):
                
                if not self._is_leased(index):
                    raise ValueError("Buffer has already been written to.", index, data)

                self.buffer[index]["data"] = data
                self.buffer[index]["written"] = True

                print("Data written to buffer {0}: \n {1} \n Buffer {0} added back to pool".format(index, data))

                if self.batched:
                    # If batched, check if the batch is complete
                    if self.batch_complete():
                        self.join()

                
            
            def make(self):
                """
                    @brief when called returns a handle that can be used to write to
                    the buffer, and a callback to release the data
                    
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
                index = self.indexPool.pop()
                while self._is_leased(index):
                    index = self.indexPool.pop()

                self.buffer[index]["leased"] = True
                return ( self.__make_callback, index )
            

            def writeout(self, data, *eventData):
                """
                    @Brief  Unpacks the tuple of data and executes __make_callback 
                    with the argumetns
                    @Param eventData:
                        (@ret from self.__make() , data_to_write)
                """
                callback, index = eventData[0]
                callback(index, data)
            


            #
            #   @Brief If buffer is not leased, and there is data available, returns the data, False otherwise. 
            #   Reading the buffer releases the memory and if pooled, it will be overwritten.
            #
            def get(self, index):
                """
                    Retreives data
                """
                if self._is_leased(index):
                    print("Buffer in use, cannot read")
                    return False
                else:
                    # Release buffer and add index back to pool
                    self.buffer[index]["released"] = True
                    self.indexPool.append(index)

                    if self.buffer[index]["written"]:
                        # Return data
                        return self.buffer[index]["data"]
                    else:
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
                        self.callback(self.buffer)

            def _is_leased(self, index):
                """
                    @brief returns true if the buffer with this index is leased
                """
                return self.buffer[index]["leased"]
            

            def batch_complete(self):
                """
                    @Brief Returns true if the batch is complete, false otherwise
                """
                # First try quick ways of checking
                if len(self.indexPool) > 0:
                    return False
                else:
                    for buf in self.buffer:
                        if not buf['written']:
                            return False
                    return True


        ### __enter__()
        print(self.args)
        print(self.kwargs)
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
        
