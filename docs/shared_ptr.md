# Shared Ptr
The purpose of the jump::shared_ptr ([defined here](../include/jump/shared_ptr.hpp)) is to provide a interopable shared_ptr implementation that auto-magically works on host and device code (when mem-copied).

## make_shared<T>()
Allows the creation of a shared pointer that can be allocated in host, or unified memory.

```
/// Make a shared pointer on the host (normal shared_ptr) 
make_shared(...);
make_shared_on(jump::memory::HOST, ...);
make_shared_on_host(...);

/// Make a shared pointer in unified memory
make_shared_on(jump::memory::UNIFIED, ...);
make_shared_on_unified(...);
```

Currently make_shared_on<T>(...) does not support creation of a class solely in device memory, mostly because generally we don't expect constructors to have the __device__ tag which causes issues during compilation. We do allow the shared_ptr class to manage a DEVICE pointer, however construction of that pointer must be handled externally. In the future we might add a make_shared_on_device<T>() to handle this separately.

## shared_ptr<T>
This type actually contains and manages the shared pointers internally.

```
/// Create a shared ptr managing a pointer on host
shared_ptr(T*);
shared_ptr(jump::memory::HOST, T*)

/// Create a shared ptr managing a pointer on device
shared_ptr(jump::memory::DEVICE, T*)

/// Create a shared ptr managing a pointer in unified memory
shared_ptr(jump::memory::UNIFIED, T*)
```

## Host Pointer
This manages a pointer in host memory, and can serve the same role as std::shared_ptr without any issues. However, it also has functionality such that the managed object can be copied to and/or then from the device.

### to_device()
Simply, to_device() will allocate appropriate space on device for the class and then mem-copy it from host to device. It is on the user to ensure that no host-specific data is used on the GPU that could cause issues. An additional feature is the ability to call to_device() on the owned object when to_device() is called on an owning shared_ptr. The to_device() in the owned object class must have the following prototype:
```
void to_device();
```

### from_device()
Similarly, from_device() allows the object (now-allocated in device memory) to be copied back to the host. Similarly from_device() can be implemented in the owned object for special functionality when syncing from the device. The from_device() in the owned object class must have the following prototype:
```
void from_device();
```



## UNIFIED Pointer
This manages a pointer in unified memory, this can be used on host or device without any issues. Further, unified pointers to not need to have to_device() or from_device() called for it to sync! However, an object allocated in unified memory might contain a shared_ptr to an object that may or may not be allocated in unified memory - in this case to_device() might still need to be called for the object to ensure it's child shared_ptr's are sync'd to device memory (the same applied with from_device()). In this particular case if to_device() and from_device() are implemented in the owned class type - they are called with gpu_ptr = nullptr.

## DEVICE Pointer
We expect this to be used the least (if at all), this manages a pointer to cuda memory and calls cudaFree when there are no more references.

This pointer can not be used except on device, and will not allow transfering from_device() or to_device().

## Interopable Operations
For the most part the container is managed through code running on the host, the functions available on host or device (interopable functions) are limited to the following:
```
T* get() const;
T* get_device() const;
T* operator->() const;
T& operator*() const;
explicit operator bool() const;
```
These are mostly functions to access the object, either by reference or direct pointer. Practically this means two things - functions expected to be interopable utilizing the shared_ptr should only call these functions (access the data). They should also be careful that the operations being called on the owned object are interopable, for example taking the object not by reference and thus calling a copy or move constructor that is not interopable will cause issues.

For more specific documentation on what each of the above functions do, see the doxygen or [shared_ptr.hpp](../../include/jcontrols/util/shared_ptr.hpp).

## device_pointer_ and block_->device_pointer
A lot of this is broken down in doxygen for the device_pointer_ class member, but this is a little bit more detail.

A key design goal for this pointer was to allow direct-copy to GPU and just work (*insert magic sparkle*). To enable this we must have a pointer to the object in device or unified memory stored directly in the shared_ptr object that lives in host memory to be copied from. All other state is stored in a control block each shared_ptr has a pointer to, this control block is initialized when the first shared_ptr to a particular object is created. Similarly to have the pointer to the object in unified or device memory exist solely in the shared_ptr object(s), this pointer would have to be created when the first shared_ptr to this particular object is created. However, we don't want to use GPU memory unless necessary which means we can't guarantee the actual space in GPU memory is needed / allocated and the pointer set until some arbitrary point later in the program.

The solution is another level of indirection, we could either allocate a pointer in unified or device memory that can be passed to each successive shared_ptr to the same object. This pointer would point to the actual reference to the object in unified or device memory, and coul dbe updated later. It would also be valid with a direct copy to the device. However, this would allocate that memory whether or not it will be used to actually store a pointer which we already stated we don't want. It would also produce another level of indirection in GPU code which would make it less efficient than the shared_ptr indirection already does. It could also complicate how reference tracking is handled in the control block and de-allocation.

A second solution (which is the one we use) is to also store a device_pointer in the control block. This could be initialized to nullptr and updated later if/when actually in use. This ensures we don't use unified or device memory unless actually needed. However, this would not directly copy to the GPU since the control block lives in host memory. This is why there must still be a direct reference in the shared_ptr still, but this can get out of sync. This is why we update the device_pointer_ from block_->device_pointer when a new shared_ptr is created to an existing object. We also provide a sync() function that can be called by higher-level code which is aware when it's about to be passed to a kernel for usage on the GPU.

A third solution which improves on the above is to allocate the control block in unified memory. However, this could lead to a slightly more complicated memory management scheme since the shared_ptr should not necessarily expect to be running with CUDA. Another down-side would be allocating more unified memory than might be necessary, which might effect performance (I don't really know how which is part of why I'm hesitant). But at a future time I might explore this.


**Last update:** *2022/07/26*
**Created:** *2022/04/26*
**Authors:** Joshua Spisak <jspisak@andrew.cmu.edu>
