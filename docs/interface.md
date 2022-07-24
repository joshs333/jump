# Interface
This is intended as early interface design scratch.

```
struct HostFunctor {

    __host__
    void operator()() {

    }
};

struct DeviceFunctor {

    __device__
    void operator()() {

    }
};


struct InteropFunctor {

    __host__ __device__
    void operator()() {

    }
};
```

```
jpc::foreach(array, functor());
jpc::foreach(multi_array, functor());

jpc::for(array, functor())
```