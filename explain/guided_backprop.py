import torch
from torch.nn import ReLU

class GuidedBackprop():
    def __init__(self, model, first_layer, grad_index):
        self.model = model
        self.gradients = None
        self.first_layer = first_layer
        self.grad_index = grad_index    
        self.forward_relu_outputs = []
        self.model.eval()
        self.update_relus()
        self.hook_layers()
    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            # for Linear output is 1, for Conv1D output is 0
            self.gradients = grad_in[self.grad_index]

        # In your GuidedBackprop's __init__ method:
        # The input layer of the encoder, which in your TCVAE is the first module in the encoder.
        for module in self.model.named_modules():
            # module[0] is name [1] is the module itself
            if module[0] == self.first_layer :
                module[1].register_backward_hook(hook_function)

    def update_relus(self):
        def relu_backward_hook_function(module, grad_in, grad_out):
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output = (corresponding_forward_output > 0).float()
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            self.forward_relu_outputs.append(ten_out)

        # Recursively hook up ReLUs in the encoder
        def register_hooks(module):
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)
            # Recurse on child modules
            for child in module.children():
                register_hooks(child)

        register_hooks(self.model.encoder)

    def generate(self, z, target_latent_dim, seq_len):
        self.model.zero_grad()
        
        # Zero out all latent dimensions except the target one
        # grad_target = torch.zeros_like(z)
        # grad_target[0][target_latent_dim] = 1.0
        grad_target= z

        # Backward pass
        z.backward(gradient=grad_target, retain_graph=True)
        
        gradients_as_arr = self.gradients.data.numpy()
        # print(gradients_as_arr.shape)
        
        return gradients_as_arr