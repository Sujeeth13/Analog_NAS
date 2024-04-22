### IBM Analog NAS 


#### Resources to learn more about VQ-VAE: 
- Reading Resource: https://jaan.io/what-is-variational-autoencoder-vae-tutorial/
- Reading Resource: https://shashank7-iitd.medium.com/understanding-vector-quantized-variational-autoencoders-vq-vae-323d710a888a
- Example Code: https://github.com/google-deepmind/sonnet/blob/v1/sonnet/examples/vqvae_example.ipynb
- VQ-VAE-2-on-Imgs: https://github.com/rosinality/vq-vae-2-pytorch
- **VQ-VAE Implementation in Pytorch**: https://nbviewer.org/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb or https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
- 


# Things to work on 
- TODO: FIX: The render arcitecture latent vectors are moving around in the plot 
- TODO: FIX: Why are there 2048 steps that the agent is taken? --> it should be ploting after each episode --> Print the number of steps taken for each episode --> move the keeping track of states from the rendercallback class to the environment
- TODO: Train different models --> Select the best performing model 
- TODO: Hyperparameter tuning for the best performing model 
- TODO: Abalation studies
- TODO: Optimize XGBOOST + Clear a more elaborate version of reward (include other accuracies)