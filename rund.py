import paddle
print(paddle.utils.run_check())           # should print True
print(paddle.device.get_device())         # should print 'gpu:0'
print(paddle.device.is_compiled_with_cuda())  # should print True
