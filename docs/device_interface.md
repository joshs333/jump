# Device Interface
The [device interface](../include/jump/device_interface.hpp) defines a nice collection of macros / functions to perform compile-time and run-time evaluation of device availability. The jump::device_interface class also defines functions to constexpr evaluate device compatibility / interfacing (eg: if there is a function to call to setup a class for operation on device, eg: do member memory transfer, etc.)

Currently we do static linking with CUDA RT which allows us to not worry about dynamic library loading, but there is code commented out there that allows runtime loading of cudart to query the number of GPUs. There was some reason (I can't remember) that I thought that was going to be useful... hmmm.
