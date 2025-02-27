import torch
import math
import matplotlib.pyplot as plt
import random
import numpy as np
from src.args import parse_arguments
from src.eval import eval_single_dataset
from src.modeling import get_model
from accelerate import Accelerator
import copy
from tqdm import tqdm
import time
from src.datasets.registry import get_dataset

# a = (0, 0)
# b = (0.5, 1)
# c = (1, 0)

def get_abc(xi, yi,s=.5):
    a = 1-yi*s-xi
    b = yi
    c = xi-yi*s
    return a, b, c

args = parse_arguments()
if args.datasets is None:
    dataset = "Cars"
else:
    dataset = args.datasets[0]
print(dataset)
args.train_dataset = dataset

vision="_vis"
if args.full_clip:
    vision = ""

#Set the granularity of the grid here, the higher n the less points are interpolated, the faster the compute
n = 10
#X = np.arange(-.5, 1.5, 1/n)
#Y = np.arange(-.5, 2, 1/n)
X = np.arange(-1, 1.5, 1/n)
Y = np.arange(-.5, 2, 1/n)

if True:
    merged_state_dict = []

    norms = []

    assert len(args.vis_models) == 3
    merged_state_dict = []
    for j, w in enumerate(args.vis_models[:3]):
        model = torch.load(w, map_location='cpu')
        model.eval()#Merge
        merged_state_dict.append(copy.deepcopy({k:v for k,v in model.state_dict().items() if 'lora' not in k}))

    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=args.num_grad_accumulation)
    model = get_model(args)
    zs = copy.deepcopy(model.state_dict())

    for i in range(len(merged_state_dict)):
        merged_state_dict[i] = {k:v-zs[k] for k,v in merged_state_dict[i].items() if k in zs.keys()}
    
    losses = np.zeros((X.shape[0], Y.shape[0]))
    dataset = get_dataset(
        dataset,
        model.encoder.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    dataloader = dataset.train_loader
    model, dataloader = accelerator.prepare(model, dataloader)

    print(f"Computing loss maxima and verifying test accuracy of trained models")
    
    max_loss = 0
    model.load_state_dict(zs)
    state = {k:v + merged_state_dict[0][k] for k,v in zs.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    metrics = eval_single_dataset(model, dataloader, args, train=True)
    max_loss = max(max_loss, metrics["loss"].mean().cpu().numpy())
    print(metrics["loss"].mean().cpu().numpy())
    
    model.load_state_dict(zs)    
    state = {k:v + merged_state_dict[1][k] for k,v in zs.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    metrics = eval_single_dataset(model, dataloader, args, train=True)
    max_loss = max(max_loss, metrics["loss"].mean().cpu().numpy())
    print(metrics["loss"].mean().cpu().numpy())
    

    model.load_state_dict(zs)
    state = {k:v + merged_state_dict[2][k] for k,v in zs.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    metrics = eval_single_dataset(model, dataloader, args, train=True)
    max_loss = max(max_loss, metrics["loss"].mean().cpu().numpy())
    print(metrics["loss"].mean().cpu().numpy())

    x, y = [], []

    print(f"Evaluating interpolated model on a {len(X)}x{len(Y)} grid")
    
    for i, xi in enumerate(X):
        print(f"{i}/{len(X)}")
        for j, yi in enumerate(Y):
            model.load_state_dict(zs, strict=False)
            a, b, c = get_abc(xi, yi)

            x.append(xi)
            y.append(yi)

            state = {k:v + a*merged_state_dict[0][k] + b*merged_state_dict[1][k] + c*merged_state_dict[-1][k] for k,v in zs.items()}

            model.load_state_dict(state, strict=True)
            model.eval()

            metrics = eval_single_dataset(model, dataloader, args, train=True)

            losses[i, j] = metrics["loss"].mean().cpu().numpy()
            import gc;gc.collect();torch.cuda.empty_cache()

    torch.save((losses, max_loss), f"losses_landscape_{dataset}{args.model}.pth")
    
else:
    losses, max_loss = torch.load(f"losses_landscape_{dataset}{args.model}.pth")


from chart_studio import plotly
import plotly.graph_objs as go
from chart_studio import plotly as py
x, y = np.meshgrid(X, Y)
colorscale=[[0.0, 'rgb(20,29,67)'],
           [0.1, 'rgb(28,76,96)'],
           [0.2, 'rgb(16,125,121)'],
           [0.3, 'rgb(92,166,133)'],
           [0.4, 'rgb(182,202,175)'],
           [0.5, 'rgb(253,245,243)'],
           [0.6, 'rgb(230,183,162)'],
           [0.7, 'rgb(211,118,105)'],
           [0.8, 'rgb(174,63,95)'],
           [0.9, 'rgb(116,25,93)'],
           [1.0, 'rgb(51,13,53)']]

top = max_loss*1.2
losses[losses > top] = top
z = np.array(losses)#.transpose()
print(max_loss)
print(get_abc(0,0))

#turbid, tempo
colorscale="tempo_r"
from matplotlib.colors import LinearSegmentedColormap
#colorscale = LinearSegmentedColormap.from_list('', ['white', 'darkblue'])

trace1= go.Surface(
    x=tuple(x),
    y=tuple(y),
    z=tuple(z),
    colorscale=colorscale,
    contours = {
        "z": {"show": True, "start": losses.min(), "end": top, "size": (top-losses.min()) / 10, "color":"white"}
    },
)


axis = dict(
    showbackground=True,
    backgroundcolor="rgb(230, 230,230)",
    showgrid=False,
    zeroline=False,
    showline=False,
)

ztickvals=[z.min(), z.max()+.5]
layout = go.Layout(autosize=True,
                   scene=dict(aspectratio=dict(x=1, y=1, z=0.95)),
                   )


z_offset=(np.min(z))*np.ones(z.shape)
x_offset=np.min(X)*np.ones(z.shape)
y_offset=np.min(Y)*np.ones(z.shape)

proj_z=lambda x, y, z: z#projection in the z-direction
colorsurfz=proj_z(x,y,z)
proj_x=lambda x, y, z: x
colorsurfx=proj_z(x,y,z)
proj_y=lambda x, y, z: y
colorsurfy=proj_z(x,y,z)

textx=[['y: '+'{:0.5f}'.format(y[i][j])+'<br>z: '+'{:0.5f}'.format(z[i][j])+
        '<br>x: '+'{:0.5f}'.format(x[i][j]) for j in range(z.shape[1])]  for i in range(z.shape[0])]
texty=[['x: '+'{:0.5f}'.format(x[i][j])+'<br>z: '+'{:0.5f}'.format(z[i][j]) +
        '<br>y: '+'{:0.5f}'.format(y[i][j]) for j in range(z.shape[1])] for i in range(z.shape[0])]

tracex = go.Surface(z=list(z),
                x=list(x_offset),
                y=list(y),
                colorscale=colorscale,
                showlegend=False,
                showscale=False,
                surfacecolor=colorsurfx,
               )

tracey = go.Surface(z=list(z),
                x=list(x),
                y=list(y_offset),
                colorscale=colorscale,
                showlegend=False,
                showscale=False,
                surfacecolor=colorsurfy,
               )
tracez = go.Surface(z=list(z_offset),
                x=list(x),
                y=list(y),
                colorscale=colorscale,
                showlegend=False,
                showscale=False,
                surfacecolor=colorsurfx,
               )

data=[trace1]#, tracex, tracey]
fig = go.Figure(data=data, layout=layout)
fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))
fig.update_layout(
    showlegend=False,
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
)
fig.update_scenes(zaxis_title_text='Loss')
fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
)

import plotly.offline as py
py.plot(fig)

layout = go.Layout(autosize=False,
                   width=1800,
                   height=1200,
                   scene=dict(xaxis=dict(range=[min(X), max(X)], showticklabels=False),
                              yaxis=dict(range=[min(Y), max(Y)], showticklabels=False),
                              aspectratio=dict(x=1, y=1)),                              
                   plot_bgcolor='white',
                   showlegend=False
                   )

fig = go.Figure(data =
                go.Contour(
                    z=losses,
                    line_smoothing=0.85,
                    #contours_coloring='lines',
                    colorscale=colorscale,
                    line_width=2,
                ), layout=layout)

fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.update_traces(ncontours=10, selector=dict(type='contour'))

x_0 = int(- min(X) * n)
x_1 = int((1 - min(X)) * n)
x_5 = int((.5 - min(X)) * n)
y_0 = int(- min(Y) * n)
y_5 = int((.5 - min(X)) * n)
fig.update_layout(
    font=dict(
        size=50,  # Set the font size here
        color="black"
    )
)
#fig.add_scatter(x=[x_0, x_1, x_5],y=[y_0, y_0, y_5],mode='markers',showlegend=False,marker={'size':10})

time.sleep(1)
py.plot(fig)
