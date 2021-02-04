

from rl_utils import reps
import numpy as np 
import matplotlib.pyplot as plt 
import sklearn.gaussian_process as gp
from scipy.stats import multivariate_normal
import sklearn
import copy 

import sys
import os
import pprint
import torch.nn as nn
# from torchvision.transforms import functional as F
import torch.optim as optim
import gpytorch
import torch
import time
import json
import copy
from tqdm import tqdm
from sklearn.metrics import f1_score
from scipy.stats import norm

from policy_learner import REPSPolicyLearner

     
class RewardLearner:
    '''Reward_Learner for ARL using GPytorch for GP implementation 
    '''
    def __init__(self, kappa):
        self.kappa = kappa #EPD sampling threshold

    def plot_rewardModel_vs_oracle_rewards(self, reward_model_rewards_all_mean_buffer, automated_expert_rewards_all):
        plt.plot(reward_model_rewards_all_mean_buffer)
        plt.plot(automated_expert_rewards_all)
        plt.title('expert rewards vs reward model rewards')
        plt.legend(('GP reward model rewards', 'oracle expert rewards'))
        plt.show()

    def remove_already_queried_samples_from_list(self, samples_to_query, queried_samples_all):
        num_samples = len(samples_to_query)
        samples_to_query_new = copy.deepcopy(samples_to_query)        
        for i in range(0, num_samples):
            if samples_to_query[i] in queried_samples_all:
                samples_to_query_new.remove(samples_to_query[i])
        return samples_to_query_new

    def convert_list_outcomes_to_array(self, outcomes_list):
        import pdb; pdb.set_trace()
        num_samples = len(outcomes_list)
        num_features = len(outcomes_list[0])
        outcomes_arr = np.zeros((num_samples, num_features))
        for i in range(0,len(outcomes_list)):
            outcomes_arr[i,:] = np.array(outcomes_list[i])
        return outcomes_arr

    def compute_KL_div_sampling_updated(self, agent, num_samples, pi_tilda_mean, pi_tilda_cov, \
        pi_star_mean, pi_star_cov, pi_current_mean, pi_current_cov, initial_wts, cut_type, S): #taking samples from policies pi_star and pi_tilda
        sampled_params_pi_tilda, sampled_params_pi_star, sampled_params_pi_current = [], [], []

        # print('computing KL div')
        # sample params from each of the three policies
        # import pdb; pdb.set_trace()

        # ADDING TO DEBUG NUM INSTAB
        pi_tilda_cov = pi_tilda_cov + np.eye(9)*5
        pi_star_cov = pi_star_cov + np.eye(9)*5

        for i in range(0, num_samples):
            new_params_pi_tilda = agent.sample_new_params_from_policy_only_mu_sigma(pi_tilda_mean, pi_tilda_cov, initial_wts, cut_type, S)
            new_params_pi_star = agent.sample_new_params_from_policy_only_mu_sigma(pi_star_mean, pi_star_cov, initial_wts, cut_type, S)
            new_params_pi_current = agent.sample_new_params_from_policy_only_mu_sigma(pi_current_mean, pi_current_cov, initial_wts, cut_type, S)

            sampled_params_pi_tilda.append(new_params_pi_tilda)
            sampled_params_pi_star.append(new_params_pi_star)
            sampled_params_pi_current.append(new_params_pi_current)

        sampled_params_pi_tilda = np.array(sampled_params_pi_tilda)
        sampled_params_pi_star = np.array(sampled_params_pi_star)
        sampled_params_pi_current = np.array(sampled_params_pi_current)

        pi_star_wi = multivariate_normal.pdf(sampled_params_pi_star, mean=pi_star_mean, cov=pi_star_cov, allow_singular=True)
        pi_tilda_wi = multivariate_normal.pdf(sampled_params_pi_tilda, mean=pi_tilda_mean, cov=pi_tilda_cov, allow_singular=True)
        pi_current_wi = multivariate_normal.pdf(sampled_params_pi_current, mean=pi_current_mean, cov=pi_current_cov, allow_singular=True)

        star_div_current = (pi_star_wi/pi_current_wi)
        star_div_tilda = (pi_star_wi/pi_tilda_wi)

        # import pdb; pdb.set_trace()
        #approx_sampling_KLdiv = (1/num_samples)*np.sum(star_div_current*np.log(star_div_tilda))
        #approx_sampling_KLdiv = (1/num_samples)*np.sum(star_div_current*(np.log(pi_star_wi) - np.log(pi_tilda_wi)))

        # removed pi_current
        approx_sampling_KLdiv = (1/num_samples)*np.sum(star_div_tilda*np.log(star_div_tilda))

        import pdb; pdb.set_trace()
        return approx_sampling_KLdiv
   
           
    def train_GPmodel(self, work_dir, num_epochs, optimizer, model, likelihood, mll, train_x, train_y):
        self.work_dir = work_dir 
        print('training model')    
        #import pdb; pdb.set_trace()
        model.train()
        likelihood.train()
        for epoch in range(num_epochs):
            # Zero backprop gradients
            optimizer.zero_grad()
            # Get output from model
            output = model(train_x) #output.mean and output.variance returns mean and var of model
            # Calc loss and backprop derivatives
            loss = -mll(output, train_y)        
            loss.backward()

            # only print this info if we're training/updating GP model, not when computing EPD
            if num_epochs > 10:
                print('GP Model Training Epoch%d, Loss:%.3f, scale:%.3f' % (epoch, loss.item(), model.covar_module.outputscale.item()))
                # print updated covar matrix 
                if epoch % 20 == 0:
                #print('updated covariance matrix', output.lazy_covariance_matrix.evaluate())
                #print('model noise', model.likelihood.noise.item())
                # save updated covariance_matrix
                    covmat_np = output.lazy_covariance_matrix.evaluate().detach().numpy()
                    np.savetxt(work_dir + '/' + 'GP_reward_model_data' + '/' + 'GP_cov_mat/' + 'epoch%i_numTrainSamples%i.txt'%(epoch, train_x.shape[0]), covmat_np)

            optimizer.step() #updates lengthscale, signal variance, AND g NN weights
        print('updated lengthscale: ', model.covar_module.base_kernel.lengthscale)
        print('updated outputscale: ', model.covar_module.outputscale)
        print('updated covariance matrix', output.lazy_covariance_matrix.evaluate())
        print('done training')

        self.num_reward_features = model.num_features
        return model

    def calc_expected_reward_for_observed_outcome_w_GPmodel(self, model, likelihood, new_outcomes):
        # import pdb; pdb.set_trace()
        # convert new_outcomes data to torch tensor
        if type(new_outcomes)==np.ndarray: 
            if len(new_outcomes.shape) == 1:
                new_outcomes =  np.expand_dims(new_outcomes, axis=0) # expand dims to be 1xn
            new_outcomes = torch.from_numpy(new_outcomes)
            new_outcomes = new_outcomes.float()
        model.eval()
        likelihood.eval()
        print('evaluating model')
        mean_expected_rewards, var_expected_rewards =[], [] 
        with torch.no_grad(), gpytorch.settings.use_toeplitz(False):              
            preds = model(new_outcomes)
            mean_expected_rewards = preds.mean.numpy().tolist()
            var_expected_rewards = preds.variance.numpy().tolist()
        
        # import pdb; pdb.set_trace()
        return mean_expected_rewards, var_expected_rewards
    
    def calc_PI_all_outcomes(self, prior_training_data, queried_samples_all, lambda_thresh, eps=0.01, beta = 0.5):
        prior_training_data_expect_rewards_mean, prior_training_data_policy_params, \
            prior_training_data_expect_rewards_sig = [], [], []        
        prior_training_data_o = np.empty([0,self.num_reward_features])
        
        for i in range(len(prior_training_data)):
            prior_training_data_o = np.vstack((prior_training_data_o, prior_training_data[i][0]))
            prior_training_data_expect_rewards_mean.append(prior_training_data[i][1])
            prior_training_data_expect_rewards_sig.append(prior_training_data[i][2])
            prior_training_data_policy_params.append(prior_training_data[i][3])      
       
        prior_training_data_expect_rewards_mean = np.array(prior_training_data_expect_rewards_mean)
        prior_training_data_expect_rewards_sig = np.sqrt(np.array(prior_training_data_expect_rewards_sig)) # NOTE: add sqrt to get std from variance!!
        prior_training_data_policy_params = np.array(prior_training_data_policy_params)

        import pdb; pdb.set_trace()
        num_samples = len(prior_training_data)       
        samples_to_query, PI_o_all = [], []
        f_oStar = np.max(prior_training_data_expect_rewards_mean)
        
        # import pdb; pdb.set_trace()
        for i in range(0, num_samples):       
            mu_o = prior_training_data_expect_rewards_mean[i]
            sigma_o = prior_training_data_expect_rewards_sig[i]  
            PI_o = norm.cdf((mu_o - f_oStar - eps)/sigma_o)
            PI_o_all.append(PI_o)
        
        max_PI_o = np.max(PI_o_all)
        max_PI_outcome_idx = np.argmax(PI_o_all) 
        import pdb; pdb.set_trace()

        if max_PI_outcome_idx not in queried_samples_all:         
            uncertaint_thresh =  prior_training_data_expect_rewards_sig[max_PI_outcome_idx]/np.sqrt(beta)
            if uncertaint_thresh > lambda_thresh:
                samples_to_query.append(max_PI_outcome_idx)

        queried_outcomes_arr = prior_training_data_o[samples_to_query]  
        print('samples_to_query', samples_to_query)            
        import pdb; pdb.set_trace()
        return samples_to_query, queried_outcomes_arr
  

    def compute_EPD_for_each_sample_updated(self, current_epoch, num_samples_each_epoch, work_dir, num_training_epochs, optimizer, current_reward_model, likelihood, mll, \
            agent, pi_tilda_mean, pi_tilda_cov, pi_current_mean, pi_current_cov, prior_training_data, \
                queried_samples_all, GP_training_data_x_all, GP_training_data_y_all, beta, initial_wts, cut_type, S):

        agent = REPSPolicyLearner()        

        prior_training_data_expect_rewards_mean, prior_training_data_policy_params, \
            prior_training_data_expect_rewards_sig = [], [], []
        
        prior_training_data_o = np.empty([0,self.num_reward_features])
        
        for i in range(len(prior_training_data)):
            prior_training_data_o = np.vstack((prior_training_data_o, prior_training_data[i][0]))
            prior_training_data_expect_rewards_mean.append(prior_training_data[i][1])
            prior_training_data_expect_rewards_sig.append(prior_training_data[i][2])
            prior_training_data_policy_params.append(prior_training_data[i][3])      
       
        prior_training_data_expect_rewards_mean = np.array(prior_training_data_expect_rewards_mean)
        prior_training_data_expect_rewards_sig = np.sqrt(np.array(prior_training_data_expect_rewards_sig)) # NOTE: add sqrt to get std from variance!!
        prior_training_data_policy_params = np.array(prior_training_data_policy_params)

        import pdb; pdb.set_trace()
        '''
        prior_training_data_o: nx7 arr (7 reward features)
        prior_training_data_expect_rewards_mean: (n,) arr
        prior_training_data_expect_rewards_sig: (n,) arr
        prior_training_data_policy_params: nx8 arr
        '''

        num_samples = len(prior_training_data)       
        samples_to_query, KL_div_all, KL_div_all_wts = [], [], []

        if current_epoch == 1:
            start_sample_to_check = num_samples_each_epoch        
        elif current_epoch == 2:
            start_sample_to_check = 2*num_samples_each_epoch   
        elif current_epoch == 3:
            start_sample_to_check = 3*num_samples_each_epoch  

        # iterate through all samples in current epoch in training data set
        for i in range(0, num_samples):       
            # don't iterate through all samples, skip already queried and ones from previous rollouts (?)     
            if i in queried_samples_all: # total_samples - samples_in_current_epoch
                continue            
            else:
                outcome = np.expand_dims(prior_training_data_o[i,:],axis=0)
                #outcome = np.expand_dims(prior_training_data_o[i,:,:,:,:],axis=0)
                mean_expect_reward = prior_training_data_expect_rewards_mean[i]
                sigma_expect_reward = prior_training_data_expect_rewards_sig[i]            

                sigma_pt_1 = mean_expect_reward + sigma_expect_reward
                sigma_pt_2 = mean_expect_reward - sigma_expect_reward

                #### SHOULD be using sigma points to estimate UPDATED reward model!
                # TODO: should be updating reward model separately for each sigma pt??
                outcomes_to_update = np.vstack((outcome, outcome))
                rewards_to_update = np.array([sigma_pt_1, sigma_pt_2])
                
                #updating hypoth_reward_model for this sample instead of actual model           
                hypoth_reward_model = copy.deepcopy(current_reward_model)
                hypoth_likelihood = copy.deepcopy(likelihood)
                hypoth_optimizer = copy.deepcopy(optimizer) #TODO: need to redefine this using hypoth_reward_model?
                hypoth_mll = copy.deepcopy(mll)            
                
                # GP_training_data_x_all and GP_training_data_y_all are previous training data for 
                og_train_x = copy.deepcopy(GP_training_data_x_all)
                og_train_y = copy.deepcopy(GP_training_data_y_all)            
                updated_train_x = np.vstack((og_train_x, outcomes_to_update))
                updated_train_y = np.concatenate((og_train_y, rewards_to_update))            
                
                # NOTE: MIGHT need to update likelihood here b/c of added noise params 
                #hypoth_likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.ones(updated_train_x.shape[0]) * beta)

                #update hypoth reward model with this outcome
                continue_training = False
                hypoth_reward_model = self.update_reward_GPmodel(work_dir, continue_training, num_training_epochs, hypoth_optimizer, hypoth_reward_model, hypoth_likelihood, hypoth_mll, updated_train_x, updated_train_y)
                
                #calculate rewards for training data under updated reward model                   
                mean_exp_rewards, var_exp_rewards = self.calc_expected_reward_for_observed_outcome_w_GPmodel(hypoth_reward_model, \
                    hypoth_likelihood, prior_training_data_o)
                # import pdb; pdb.set_trace()
                #Calculate policy update under updated reward model            
                # NOTE: 2/4/21 update: lower rel_entropy_bound so new policy cov doesn't deviate too far 
                pi_star_mean, pi_star_cov, reps_wts = agent.update_policy_REPS(mean_exp_rewards, \
                    prior_training_data_policy_params, rel_entropy_bound = 0.4, min_temperature=0.001) 
                pi_star_wts = reps_wts

                print('pi_current_mean (policy before updating)' , pi_current_mean)
                print('pi_tilda_mean (new policy under current reward model)' , pi_tilda_mean)
                print('pi_star_mean (new policy under updated reward model)' , pi_star_mean)

                # save policy mean and covs for debugging
                np.savez(work_dir + '/' + 'GP_reward_model_data/policy_pi_star_tilda_data/' + 'epoch_' + str(current_epoch) + '_pi_star_tilda_sample_' + str(i) + '.npz', 
                    pi_current_mean = pi_current_mean, pi_tilda_mean = pi_tilda_mean, pi_star_mean = pi_star_mean,
                        pi_current_cov = pi_current_cov, pi_tilda_cov = pi_tilda_cov, pi_star_cov = pi_star_cov)
                # import pdb; pdb.set_trace()

                # note - 40 is higher # samples (used to be 10)
                print('computing KL div')
                # import pdb; pdb.set_trace()
                KL_div = self.compute_KL_div_sampling_updated(agent, 40, pi_tilda_mean, pi_tilda_cov, \
                    pi_star_mean, pi_star_cov, pi_current_mean, pi_current_cov, initial_wts, cut_type, S)
                
                print('KLdiv_sampling', KL_div)
                KL_div_all.append(KL_div)                    
                     
                if (np.all(np.isnan(KL_div)==True))==False and np.any(KL_div >= self.kappa):
                    samples_to_query.append(i)

                # TODO: TRY calculating KL div based on weights (avoid num instabilities?)
                # KL_div_wts = np.sum()
                # print('KLdiv_wts', KL_div_wts)
                # KL_div_all_wts.append(KL_div_wts) 

        #Check if we've already queried these samples. If yes, remove from list:
        # import pdb; pdb.set_trace()
        print('KL divs', KL_div_all)
        print('median KL DIV', np.median(KL_div_all))
        samples_to_query_new = self.remove_already_queried_samples_from_list(samples_to_query, queried_samples_all)
        print('new samples_to_query', samples_to_query_new)
        import pdb; pdb.set_trace()
        queried_outcomes_arr = prior_training_data_o[samples_to_query_new]              
        import pdb; pdb.set_trace()
        return samples_to_query_new, queried_outcomes_arr #indexes of samples to query from expert
    
    def update_reward_GPmodel(self, work_dir, continue_training, num_training_epochs, optimizer, model, likelihood, mll, updated_train_x, updated_train_y):
        # if updated_train data are np arrays, convert to torch float tensors
        #import pdb; pdb.set_trace()
        if type(updated_train_x)==np.ndarray:
            updated_train_x = torch.from_numpy(updated_train_x)
            updated_train_x = updated_train_x.float()

        if type(updated_train_y)==np.ndarray:
            updated_train_y = torch.from_numpy(updated_train_y)
            updated_train_y = updated_train_y.float()
        
        model.set_train_data(inputs = updated_train_x, targets = updated_train_y, strict = False)

        epochs_to_cont_training = num_training_epochs        
        if epochs_to_cont_training != 0:
            model = self.train_GPmodel(work_dir, epochs_to_cont_training, optimizer, model, likelihood, mll, updated_train_x, updated_train_y)
        
        return model
    
    def calc_mean_std_reward_features(self, cut_type, save_mean_std = False):
        '''
        save_mean_std: True, False
        cut_type: normal, pivchop, scoring

        this function calculates the mean and std from prior training data and saves to an array for loading in the future
        '''
        base_dir = '/home/sony/Documents/cutting_RL_experiments/data/Jan-2021-LL-param-exps/' + cut_type + '/'
        food_types = ['potato', 'celery', 'carrot', 'banana', 'tomato', 'mozz']
        reward_feats_all_foods = np.empty((0,7))
        for food in food_types:
            #import pdb; pdb.set_trace()
            work_dir = base_dir + food + '/'
            exp_folder = os.listdir(work_dir)
            work_dir = work_dir + exp_folder[0] + '/'

            reward_feats = np.load(work_dir + '/reward_features_all_samples.npy')
            # import pdb; pdb.set_trace()
            if len(np.where(reward_feats==np.inf)[0])!= 0: # there are inf's in array
                inf_inds = np.unique(np.where(reward_feats==np.inf)[0])
                last_inf_ind = inf_inds.max()
                reward_feats = reward_feats[last_inf_ind+1:]
            reward_feats_all_foods = np.vstack((reward_feats_all_foods, reward_feats))
        
        #import pdb; pdb.set_trace()
        reward_feats_mean = np.mean(reward_feats_all_foods, axis=0)
        reward_feats_std = np.std(reward_feats_all_foods, axis=0)

        if save_mean_std:
            np.save(base_dir + '/' + cut_type + '_reward_feats_mean_std' + '.npy', np.array([reward_feats_mean, reward_feats_std]))
        
        return reward_feats_mean, reward_feats_std

    def standardize_reward_feature(self, cut_type, current_reward_feat):
        '''
        standardize reward features using mean and std from previous training data
        '''
        base_dir = '/home/sony/Documents/cutting_RL_experiments/data/Jan-2021-LL-param-exps/' + cut_type + '/'
        mean_std_reward_feats = np.load(base_dir + '/' + cut_type + '_reward_feats_mean_std' + '.npy')
        mean_reward_feats = mean_std_reward_feats[0,:]
        std_reward_feats = mean_std_reward_feats[1,:]
        #import pdb; pdb.set_trace()
        standardized_reward_feat = (current_reward_feat - mean_reward_feats)/std_reward_feats

        return standardized_reward_feat

    def unstandardize_reward_feature(self, cut_type, standardized_reward_feat):
        base_dir = '/home/sony/Documents/cutting_RL_experiments/data/Jan-2021-LL-param-exps/' + cut_type + '/'
        mean_std_reward_feats = np.load(base_dir + '/' + cut_type + '_reward_feats_mean_std' + '.npy')
        mean_reward_feats = mean_std_reward_feats[0,:]
        std_reward_feats = mean_std_reward_feats[1,:]

        unstandardized_reward_feat = (standardized_reward_feat*std_reward_feats) + mean_reward_feats

        return unstandardized_reward_feat

   
# class Reward_Learner:
#     '''Reward_Learner for original block grasping task
#     '''
#     def __init__(self, lambd, beta, kappa):
#         self.lambd = lambd #tuning parameter that affects how often agent queries the expert for reward
#         self.beta = beta #noise added to observations (to model human imprecision)
#         self.kappa = kappa #EPD sampling threshold

#     def remove_already_queried_samples_from_list(self, samples_to_query, queried_samples_all):
#         num_samples = len(samples_to_query)
#         samples_to_query_new = copy.deepcopy(samples_to_query)
        
#         for i in range(0, num_samples):
#             #import pdb; pdb.set_trace()
#             if samples_to_query[i] in queried_samples_all:
#                 samples_to_query_new.remove(samples_to_query[i])
#         return samples_to_query_new

#     def convert_list_outcomes_to_array(self, outcomes_list):
#         num_samples = len(outcomes_list)
#         num_features = len(outcomes_list[0])
#         outcomes_arr = np.zeros((num_samples, num_features))
#         for i in range(0,len(outcomes_list)):
#             outcomes_arr[i,:] = np.array(outcomes_list[i])
#         return outcomes_arr


#     def compute_kl_divergence(self, pm, pv, qm, qv):    
#         """
#         Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
#         Also computes KL divergence from a single Gaussian pm,pv to a set
#         of Gaussians qm,qv.
#         Diagonal covariances are assumed.  Divergence is expressed in nats.
#         """
#         if (len(qm.shape) == 2):
#             axis = 1
#         else:
#             axis = 0
#         # Determinants of diagonal covariances pv, qv
#         dpv = pv.prod()
#         dqv = qv.prod(axis)
#         # Inverse of diagonal covariance qv
#         iqv = 1./qv
#         # Difference between means pm, qm
#         diff = qm - pm

#         #import pdb; pdb.set_trace()
#         return (0.5 *(np.log(dqv / dpv) + (iqv * pv).sum(axis) + (diff * iqv * diff).sum(axis) - len(pm))) 

#     def compute_KL_div_sampling(self, agent, num_samples, pi_tilda_mean, pi_tilda_cov, pi_star_mean, pi_star_cov): #taking samples from policies pi_star and pi_tilda
#         #print('computing numerical sampling-based KL_div')
#         sampled_params_pi_tilda = []
#         sampled_params_pi_star = []
#         #import pdb; pdb.set_trace()
#         for i in range(0, num_samples):
#             sampled_params_pi_tilda.append(agent.sample_params_from_policy(pi_tilda_mean, pi_tilda_cov))
#             sampled_params_pi_star.append(agent.sample_params_from_policy(pi_star_mean, pi_star_cov))

#         sampled_params_pi_tilda = np.array(sampled_params_pi_tilda)
#         sampled_params_pi_star = np.array(sampled_params_pi_star)
#         #import pdb; pdb.set_trace() 

#         div = (sampled_params_pi_star/sampled_params_pi_tilda)
#         div[div<=0]=.01
#         approx_sampling_KLdiv = (1/num_samples)*np.sum(div*np.log(div))

#         return approx_sampling_KLdiv

#     def initialize_reward_model(self, signal_var_initial, length_scale_initial):
#         print('initializing reward model')
#          #initialize GP with zero mean prior 
#         #kernel = gp.kernels.RBF(length_scale_initial, (1e-3, 1e3))

#         kernel = gp.kernels.ConstantKernel(signal_var_initial, (1, 20)) \
#             * gp.kernels.RBF(length_scale_initial, (40, 150))
#         #import pdb; pdb.set_trace()
#         gpr_reward_model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, alpha=0.0001)
#         #import pdb; pdb.set_trace()
#         return gpr_reward_model
        
#     def compute_EPD_for_each_sample(self, agent, pi_tilda_mean, pi_tilda_cov, current_reward_model,\
#         prior_training_data, queried_samples_all): #output o+ = argmax u(o)
#         agent = REPS_policy_learner() 
        
#         prior_training_data = np.array(prior_training_data)
#         prior_training_data_o = prior_training_data[:,0]
#         prior_training_data_expect_rewards_mean = prior_training_data[:,1]
#         prior_training_data_expect_rewards_sig = prior_training_data[:,2]
#         prior_training_data_policy_params = prior_training_data[:, 3]
        
#         num_samples = len(prior_training_data)       
#         #import pdb; pdb.set_trace()
#         samples_to_query = []
#         for i in range(0, num_samples):
#             outcome = prior_training_data_o[i]
#             mean_expect_reward = prior_training_data_expect_rewards_mean[i]
#             sigma_expect_reward = prior_training_data_expect_rewards_sig[i]            

#             sigma_pt_1 = mean_expect_reward + sigma_expect_reward
#             sigma_pt_2 = mean_expect_reward - sigma_expect_reward

#             #### SHOULD be using sigma points to estimate UPDATED reward model!!!!! ()
#             outcomes_to_update = np.array([outcome, outcome])
#             rewards_to_update = np.array([sigma_pt_1, sigma_pt_2])
            
#             #updating hypoth_reward_model for this sample instead of actual model           
#             hypoth_reward_model = copy.deepcopy(current_reward_model)

            
#             #update hypoth reward model with this outcome
#             hypoth_reward_model = self.update_reward_model(hypoth_reward_model, outcomes_to_update, rewards_to_update)
            
#             #calculate rewards for training data under updated reward model                   
#             #import pdb; pdb.set_trace()
#             mean_exp_rewards, var_exp_rewards = \
#                 self.calc_expected_reward_for_an_observed_outcome(agent, \
#                     hypoth_reward_model, prior_training_data_o)      
            
#             #Calculate policy update under updated reward model
#             pi_star_mean, pi_star_cov, reps_wts = agent.update_policy_REPS(mean_exp_rewards, \
#                 prior_training_data_policy_params, rel_entropy_bound=1.5, min_temperature=0.001) 
            
           
#             KL_div = self.compute_KL_div_sampling(agent, \
#                 10, pi_tilda_mean, pi_tilda_cov, pi_star_mean, pi_star_cov)
            
#             print('KLdiv_sampling', KL_div)
            
#             #Calculate KL-diverg b/w the two policies
#             # KL_div = self.compute_kl_divergence(pi_star_mean, pi_star_cov, pi_tilda_mean, pi_tilda_cov)
             
#             if (np.all(np.isnan(KL_div)==True))==False and np.any(KL_div >= self.kappa):
#                 samples_to_query.append(i)
        
#         #Check if we've already queried these samples. If yes, remove from list:
#         samples_to_query_new = self.remove_already_queried_samples_from_list(samples_to_query,\
#             queried_samples_all)
#         print('new samples_to_query', samples_to_query_new)
#         #import pdb; pdb.set_trace()

#         queried_outcomes = prior_training_data_o[samples_to_query_new]       
#         queried_outcomes_arr = self.convert_list_outcomes_to_array(queried_outcomes)

#         return samples_to_query_new, queried_outcomes_arr #indexes of samples to query from expert


#     def calc_expected_reward_for_an_observed_outcome(self, agent, gpr_reward_model, new_outcomes): #provided to policy learner
#         mean_expected_rewards, var_expected_rewards =[], []        
#         #import pdb; pdb.set_trace()
#         if type(new_outcomes[0])==np.float64: #single new outcome (1x5 list)
#             #print('outcome', new_outcomes)
#             X=np.atleast_2d(new_outcomes)
#             mean_expected_reward, var_expected_reward = gpr_reward_model.predict(X, return_std=True)
#             mean_expected_rewards.append(mean_expected_reward[0])
#             var_expected_rewards.append(var_expected_reward[0])
        
#         else: #list of new outcomes
            
#             #import pdb; pdb.set_trace()
#             for outcome in new_outcomes:
#                 #print('outcome = ', outcome)
#                 X=np.atleast_2d(outcome)
#                 mean_expected_reward, var_expected_reward = \
#                     gpr_reward_model.predict(X, return_std=True)
                
#                 mean_expected_rewards.append(mean_expected_reward[0][0])
#                 var_expected_rewards.append(var_expected_reward[0])

#         return mean_expected_rewards, var_expected_rewards

#     def update_reward_model(self, gpr_reward_model, outcomes, rewards): #update GP for reward model p(R|o,D)
#         outcomes = np.atleast_2d(outcomes)
#         rewards = np.atleast_2d(rewards).T
#         #import pdb; pdb.set_trace()
#         gpr_reward_model.fit(outcomes, rewards) #fit Gaussian process regression model to data
#         return gpr_reward_model

#     #def plot_GRP_reward_function(self,):

    def plot_kernel_length_scale(self,epoch,gpr_reward_model):
        #print('kernel length scale', gpr_reward_model.kernel_.get_params()['length_scale'])
        length_scale = gpr_reward_model.kernel_.get_params()['k2__length_scale']
        signal_var = gpr_reward_model.kernel_.get_params()['k1__constant_value']
        print('kernel length scale', length_scale)
        print('kernel signal variance', signal_var)

        #plt.figure()c
        plt.scatter(epoch, length_scale, color='green')
        #plt.title('kernel length scale vs epochs')
        #plt.xlabel('num epochs')
        #plt.ylabel('length scale')
        #plt.axis([0, 21, 0, 150])
        plt.pause(0.05)

    def plot_rewards_all_episodes(self,training_data_list, kappa):
        plt.plot(np.array(training_data_list)[:,1])
        plt.plot(np.array(training_data_list)[:,3])
        plt.xlabel('total episodes')
        plt.ylabel('rewards')
        plt.title('rewards vs. episodes, kappa = %f'%kappa)
        plt.legend(('mean_expected_reward from reward learner model', 'expert reward'))
        plt.show()

    def plot_mean_rewards_each_epoch(self, epoch,mean_reward_model_rewards_all_epochs,mean_expert_rewards_all_epochs,kappa):
        plt.plot(np.arange(epoch), mean_reward_model_rewards_all_epochs)
        plt.plot(np.arange(epoch), mean_expert_rewards_all_epochs)
        plt.xlabel('epochs')
        plt.ylabel('rewards')
        plt.title('rewards vs. epochs, kappa = %f'%kappa)
        plt.legend(('mean_expected_reward from reward learner model', 'mean expert reward'))
        plt.show()

    def plot_cumulative_queried_samples_vs_epochs(self, epoch,total_queried_samples_each_epoch,kappa):
        plt.plot(np.arange(epoch), total_queried_samples_each_epoch)
        plt.xlabel('epochs')
        plt.ylabel('total (cumululative) queried samples')
        plt.title('total (cumulative) queried samples vs. epochs, kappa = %f'%kappa)
        plt.show()
