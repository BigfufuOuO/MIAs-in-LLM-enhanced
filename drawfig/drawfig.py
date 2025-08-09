import matplotlib.pyplot as plt
import os
import numpy as np

class DrawFig:
    def __init__(self, title):
        self.title = title
        self.save_path = './drawfig/'
        
        self.gpt2_params = [0.124, 0.774, 1.5]
        self.qwen2_params = [0.5, 1.5, 3, 7]
        self.deepseek_distill_params = [1.5, 7]
        self.llama_params = [1, 3]
        self.opt_params = [1.3, 2.7, 6.7]
        
        self.labels = ['GPT-2', 'Qwen2.5', 'DeepSeek-R1-Qwen-Distill', 'Llama3.2', 'Opt']
        
        self.length = [32, 64, 96, 128, 160, 192]
        self.n_neighs = [5, 15, 25, 50, 75, 100]
        self.epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
    def draw_params(self, y_multi, subtitle):
        num_fig = len(y_multi)
        fig, axes = plt.subplots(2, num_fig // 2, figsize=(9, 9), sharey=True)
        # 设置全局字体大小
        # plt.rcParams.update({'font.size': 14})
        # plt.title(self.title)
        axes = axes.flatten()
        
        for i in range(num_fig):
            axes[i].set_title(subtitle[i], fontsize=16)
            y = y_multi[i]

            makersize = 7
            axes[i].plot(self.gpt2_params, y[0], marker='o', markersize=makersize, label=self.labels[0])
            axes[i].plot(self.qwen2_params, y[1], marker='^', markersize=makersize,label=self.labels[1])
            axes[i].plot(self.deepseek_distill_params, y[2], marker='s', markersize=makersize, label=self.labels[2])
            axes[i].plot(self.llama_params, y[3], marker='o', markersize=makersize, linestyle='--', label=self.labels[3])
            axes[i].plot(self.opt_params, y[4], marker='p',markersize=makersize, label=self.labels[4])
            
            if i == 0:
                axes[i].set_ylabel('AUC', fontsize=14)
            
            if i == 3:
                axes[i].legend(fontsize=14)
            axes[i].set_xlabel('Model Size(B)', fontsize=14)
            axes[i].set_xlim(left=0)
            axes[i].set_xticks([0.5, 1.5, 3, 7])
            
            axes[i].tick_params(axis='both', which='major', labelsize=14)
            
            axes[i].grid(True)
        
        plt.tight_layout()
        # save figure
        path = os.path.join(self.save_path, self.title)
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, f'{subtitle}.png')
        plt.savefig(path, dpi=300)
        
    def draw_params_single(self, y, subtitle):
        plt.figure(figsize=(6, 6))
        
        makersize = 7
        plt.plot(self.gpt2_params, y[0], marker='o', markersize=makersize, label=self.labels[0])
        plt.plot(self.qwen2_params, y[1], marker='^', markersize=makersize,label=self.labels[1])
        plt.plot(self.deepseek_distill_params, y[2], marker='s', markersize=makersize, label=self.labels[2])
        plt.plot(self.llama_params, y[3], marker='o', markersize=makersize, linestyle='--', label=self.labels[3])
        plt.plot(self.opt_params, y[4], marker='p',markersize=makersize, label=self.labels[4])
        
        plt.ylabel('LOSS Difference', fontsize=14)
        plt.legend(fontsize=10)
        plt.xlabel('Model Size(B)', fontsize=14)
        plt.xlim(left=0)
        
        plt.tick_params(axis='both', which='major', labelsize=14)
        
        plt.grid(True)
        
        plt.tight_layout()
        # save figure
        path = os.path.join(self.save_path, self.title)
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, f'{subtitle}.png')
        plt.savefig(path, dpi=300)

    def draw_text_length(self, y_multi, models, methods):
        num_fig = len(y_multi)
        fig, axes = plt.subplots(2, num_fig // 2, figsize=(8, 11), sharey=True)
        axes = axes.flatten()
        for i in range(num_fig):
            axes[i].set_title(models[i], fontsize=16)
            
            y = y_multi[i]
            makersize = 7
            axes[i].plot(self.length, y[3], marker='o', markersize=makersize, color="#004b03", label=methods[3])
            axes[i].plot(self.length, y[2], marker='s', markersize=makersize, color="#007605", label=methods[2])
            axes[i].plot(self.length, y[0], marker='o', markersize=makersize, color="#00af07", label=methods[0])
            axes[i].plot(self.length, y[1], marker='^', markersize=makersize, color="#00dc09", label=methods[1])
            axes[i].plot(self.length, y[4], marker='p',markersize=makersize, linestyle='-.', color="red", label=methods[4])
            axes[i].plot(self.length, y[5], marker='x',markersize=makersize, linestyle='-.', color="orange", label=methods[5])
            axes[i].xaxis.set_ticks(self.length)
            
            if i == 0 or i == 1:
                axes[i].axvline(x=96, color='#0071dc', linestyle='--', alpha=0.5, linewidth=4)
            elif i == 2 or i == 3:
                axes[i].axvline(x=128, color='#0071dc', linestyle='--', alpha=0.5, linewidth=4)
            
            if i == 0:
                axes[i].set_ylabel('AUC', fontsize=14)
            
            # if i == 3:
            #     axes[i].legend(fontsize=10, loc='upper right', bbox_to_anchor=(1, 0.93))
            axes[i].set_xlabel('Text Length(tokens)', fontsize=14)
            
            axes[i].tick_params(axis='both', which='major', labelsize=14)
            
            axes[i].grid(True)
            
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center',
                   ncol=3, fontsize=12)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.13)
        # save figure
        path = os.path.join(self.save_path, self.title)
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, f'text_length.png')
        plt.savefig(path, dpi=300)
        
    def draw_text_length_inter(self, y_multi, models, methods):
        num_fig = len(y_multi)
        fig, axes = plt.subplots(1, num_fig, figsize=(12, 6))
        for i in range(num_fig):
            axes[i].set_title(models[i], fontsize=16)
            
            y = y_multi[i]
            makersize = 7
            axes[i].plot(self.length, y[3], marker='o', markersize=makersize, color="#004b03", label=methods[3])
            axes[i].plot(self.length, y[2], marker='s', markersize=makersize, color="#007605", label=methods[2])
            axes[i].plot(self.length, y[0], marker='o', markersize=makersize, color="#00af07", label=methods[0])
            axes[i].plot(self.length, y[1], marker='^', markersize=makersize, color="#00dc09", label=methods[1])
            axes[i].plot(self.length, y[4], marker='p',markersize=makersize, linestyle='-.', color="red", label=methods[4])
            axes[i].plot(self.length, y[5], marker='x',markersize=makersize, linestyle='-.', color="orange", label=methods[5])
            axes[i].xaxis.set_ticks(self.length)
            
            if i == 0 :
                axes[i].axvline(x=96, color='#0071dc', linestyle='--', alpha=0.5, linewidth=4)
            elif i == 1:
                axes[i].axvline(x=128, color='#0071dc', linestyle='--', alpha=0.5, linewidth=4)
            
            if i == 0:
                axes[i].set_ylabel('AUC', fontsize=14)
            
            if i == 1:
                axes[i].legend(fontsize=12)
            axes[i].set_xlabel('Text Length(tokens)', fontsize=14)
            
            axes[i].tick_params(axis='both', which='major', labelsize=14)
            
            axes[i].grid(True)
        
        plt.tight_layout()
        # save figure
        path = os.path.join(self.save_path, self.title)
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, f'text_length_inter.png')
        plt.savefig(path, dpi=300)
        
    def draw_neighbor_single_model(self, y_multi, labels):
        plt.figure(figsize=(6, 6))
        plt.title('Qwen2.5-1.5B', fontsize=16)
            
        y = y_multi[0]
        makersize = 7
        plt.plot(self.n_neighs, y[0], marker='o', markersize=makersize, label=labels[0])
        plt.plot(self.n_neighs, y[1], marker='^', markersize=makersize, label=labels[1])
        plt.plot(self.n_neighs, y[2], marker='s', markersize=makersize, label=labels[2])
            
        plt.ylabel('AUC', fontsize=14)
    
        plt.legend(fontsize=12, loc='upper right', bbox_to_anchor=(1, 0.75))
        plt.xlabel('Number of Neighbors', fontsize=14)
        
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.xticks([5, 15, 25, 50, 75, 100])
        
        plt.grid(True)
        
        plt.tight_layout()
        # save figure
        path = os.path.join(self.save_path, self.title)
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, f'neighbor_single_model.png')
        plt.savefig(path, dpi=300)
        
    def draw_neighbor_param_diff(self, y_multi, labels):
        plt.figure(figsize=(6, 6))
        plt.title('Qwen2.5', fontsize=16)
            
        y = y_multi[0]
        makersize = 7
        plt.plot(self.n_neighs, y[0], marker='o', markersize=makersize, label=labels[0])
        plt.plot(self.n_neighs, y[1], marker='^', markersize=makersize, label=labels[1])
        plt.plot(self.n_neighs, y[2], marker='s', markersize=makersize, label=labels[2])
        plt.plot(self.n_neighs, y[3], marker='p', markersize=makersize, label=labels[3])
            
        plt.ylabel('AUC', fontsize=14)
    
        plt.legend(fontsize=12, loc='upper right')
        plt.xlabel('Number of Neighbors', fontsize=14)
        
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.xticks([5, 15, 25, 50, 75, 100])
        
        plt.grid(True)
        
        plt.tight_layout()
        # save figure
        path = os.path.join(self.save_path, self.title)
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, f'neighbor_param_diff.png')
        plt.savefig(path, dpi=300)
        
    def draw_loss_diff(self, y_multi, labels):
        plt.figure(figsize=(6, 4))
        plt.title('Finetuning Qwen2.5-1.5B', fontsize=16)
            
        y = y_multi[0]
        makersize = 7
        plt.plot(self.epochs, y[0], marker='o', markersize=makersize, label=labels[0])
        plt.plot(self.epochs, y[1], marker='^', markersize=makersize, label=labels[1])
        # plt.plot(self.epochs, y[2], marker='s', markersize=makersize, label=labels[2])
            
        plt.ylabel('Loss', fontsize=14)
        offset = 2.3
        alpha = 3
        def forward(x): return (x - offset) ** alpha
        def inverse(x): 
            x = np.maximum(x, 0)
            return (x ** (1 / alpha)) + offset
        plt.yscale('function', functions=(forward, inverse))
        plt.ylim(2)
        plt.yticks([2.3, 2.65, 2.8, 3.0])
        plt.axvline(x=4, color='r', linestyle='--',)
        plt.fill_between(self.epochs[3:], 0, max(max(y[0]), max(y[1])) + 1, color='gray', alpha=0.2)
        plt.text(7, 2.9, 'Overfitting', fontsize=12, color='r', ha='center', alpha=0.8)
    
        plt.legend(fontsize=12, )
        plt.xlabel('Epochs', fontsize=14)
        
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.xticks(self.epochs)
        
        plt.grid(True)
        
        plt.tight_layout()
        # save figure
        path = os.path.join(self.save_path, self.title)
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, f'loss_diff.png')
        plt.savefig(path, dpi=300)
        
    def draw_loss_phase(self, y_multi, labels):
        plt.figure(figsize=(6, 6))
            
        y = y_multi[0]
        makersize = 7
        plt.plot(self.epochs, y[0], marker='o', markersize=makersize, label=labels[0])
        plt.plot(self.epochs, y[1], marker='^', markersize=makersize, label=labels[1])
        plt.plot(self.epochs, y[2], marker='o', markersize=makersize, label=labels[2])
        plt.plot(self.epochs, y[3], marker='s', markersize=makersize, label=labels[3])
        plt.plot(self.epochs, y[4], marker='x', linestyle='-.', markersize=makersize, label=labels[4])
        plt.plot(self.epochs, y[5], marker='p', linestyle='-.',markersize=makersize, label=labels[5])
            
        plt.ylabel('AUC', fontsize=14)
    
        plt.legend(fontsize=12,)
        plt.xlabel('Epochs', fontsize=14)
        plt.xticks(self.epochs)
        
        plt.tick_params(axis='both', which='major', labelsize=14)
        
        plt.grid(True)
        
        plt.tight_layout()
        # save figure
        path = os.path.join(self.save_path, self.title)
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, f'loss_phase.png')
        plt.savefig(path, dpi=300)