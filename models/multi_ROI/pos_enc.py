import jittor
import numpy as np
from jittor import Var


class PositionalEncoding(jittor.nn.Module):
    """
    Implement  positional encoding
    """

    def __init__(self, num_freq=32, input_dim=3, freq_factor=np.pi, include_input=True, is_pos_enc=True):
        super().__init__()
        self.num_freq = num_freq
        self.input_dim = input_dim
        # [pi*2^0,pi*2^1.......]
        self.freq = freq_factor * (2.0 ** jittor.arange(0, num_freq))  # [num_freq=32,]
        if is_pos_enc:
            self.output_dim = self.num_freq * 2 * input_dim
            self.include_input = include_input
            if include_input:
                self.output_dim += input_dim
        else:
            self.d_out = 0

        # [1,64,1]
        # [pi,pi,  2^1*pi,2^1*pi,  2^2*pi,2^1*pi......]
        self._freqs = self.freq.reshape(-1, 1).repeat(1, 2).reshape(-1).view(1, -1, 1)

        # 0 pi/2 0 pi/2 ... so that
        # all odd index = pi/2
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = jittor.zeros(2 * self.num_freq)
        _phases[1::2] = np.pi * 0.5
        self._phases = _phases.reshape(1, -1, 1)  # [1,2*num_freq,1]

    def execute(self, x):
        """
        Args:
            x:[B,input_dim]
        Returns:
            [B,output_dim]
        """
        # [B,3] => []
        embed = x.unsqueeze(1).repeat(1, self.num_freq * 2, 1)  # [B,num_freq*2,input_dim]
        # self._phases + (embed * self._freq)
        # [1,64,1] + [B,64,input_dim] * [1,64,1]
        embed = jittor.sin(add_mul(self._phases, embed, self._freqs))
        embed = embed.view(x.shape[0], -1)  # [B, num_freq*2*input_dim]
        if self.include_input:
            embed = jittor.concat((x, embed), dim=-1)  # [B, output_dim=32*2*3+3=195]
        return embed


def add_mul(inp: Var, tensor1: Var, tensor2: Var, value=1.0):
    return inp + value * (tensor1 * tensor2)

