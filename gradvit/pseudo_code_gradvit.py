#Model weight fixed
gt = dataloader()
input=torch.rand(gt.size()).requires_grad(True)

opt = torch.optimizer,SGD([input])
output = model(gt)

image_prior = func1(gt,model)
patch_prior = func2(gt,model)
grad_prior = func3(gt,model) #hook to extract backward gradients


for i in range(max_iters):
  noise_output = model(input)
  
  image_prior_input = func1(input,model)
  patch_prior_input = func2(input,model)
  grad_prior_input = func3(input,model) #hook to extract backward gradients
  
  L = torch.MAE(image_prior-image_prior_input) + torch.MAE(patch_prior-patch_prior_input) + torch.MAE(grad_prior-image_prior_input) 
  
  torch.zero_grad()
  L.backward()
  opt.step()
  
save_image(input)

