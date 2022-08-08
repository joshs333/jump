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
// for each element in the array regardless of dimensionality
jpc::foreach(array, functor());
jpc::foreach(multi_array, functor());

// For 
jpc::for(array, functor())
```



# Cuda compilation / detection
The following is a brain dump thinking about cmake structure / install requirements and trying to be more portable.

We support the following development / deployment setups:
**A**
- develop against jump source
- build on machine with no GPU / CUDA
- deploy on machine with no GPU / CUDA

**B**
- develop against jump source
- build on machine with GPU / CUDA
- deploy on machine with no GPU / CUDA
- deploy on machine with GPU / CUDA

**C**
- develop against apt installed jump
  ^^ settings depend on whether this is header-only or not?
  ^^ settings / interface should not be different if it is apt installed / against source???

I guess the question is whether the target application will be bound by 'compile' settings in the source (apt versus against git-cloned source)

So like if we package jump with CUDA enabled does it require CUDA to be installed? This is a hard nope- I'd not like to do this.

I think we provide a bare-bones default targdet.

Then provide macros to find packages / includes sorta like thrust does? Or do we assume if you are developing then you are going to clone from git? Would be nice to get some apt packages up for convenience.

I do think I can structure the cmake setup nicely later - just so long as the code is setup well. Do I want to assume CUDA by default?

This takes be back to early thoughts about jcontrols structuring using CUDA functions - cpp vs cu. Definitely reaching a point here where I feel like I should start coding and can re-structure later.

So key points:
- don't let files run cuda functions if cuda is not available at build time... once linked it's fine, but don't let those functions call unless cuda is installed on the machine.
- if we are building with cuda we assume cudart is available even in non-cuda files (since we are header only we really need this)
