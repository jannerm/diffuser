import torch
import torch.nn as nn
import pdb

class ValueGuide(nn.Module):

	def __init__(self, reward_model, value_model, discount):
		super().__init__()
		self.reward_model = reward_model
		self.value_model = value_model
		self.discount = discount

	def forward(self, x, t):
		'''
			x : [ B x 1 x H x (obs_dim + act_dim) ]
		'''
		batch_size, _, horizon, joined_dim = x.shape
		nonterm_states = x[:,:,:-1].reshape(batch_size * (horizon-1), joined_dim)
		terminal_states = x[:,:,-1].reshape(batch_size, joined_dim)

		## [ B * (H - 1) ]
		t_rep = torch.repeat_interleave(t, horizon-1, dim=0)
		## [ B x (H - 1) ]
		rewards = self.reward_model(nonterm_states, t_rep).reshape(batch_size, horizon-1)
		## [ B x 1 ]
		terminal_values = self.value_model(terminal_states, t)

		## [ B x H ]
		joined = torch.cat([rewards, terminal_values], dim=-1)
		## [ H ]
		discounts = self.discount ** torch.arange(horizon, device=x.device)

		## [ B ]
		output = (joined * discounts).sum(dim=-1)
		return output