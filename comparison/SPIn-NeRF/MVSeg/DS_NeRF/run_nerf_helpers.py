import cv2
import torch
import clip
import torchvision

torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import searchsorted

from matplotlib import pyplot as plt

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears + 1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear + 1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears + 1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear + 1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear + 1]))


class NeRF_RGB(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False,
                 alpha_model=None):
        """ 
        """
        super(NeRF_RGB, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            # self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

        self.alpha_model = alpha_model

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            with torch.no_grad():
                alpha = self.alpha_model(x)[..., 3][..., None]
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears + 1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear + 1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears + 1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear + 1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear + 1]))


# Ray helpers
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                       -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32),
                       indexing='xy')  # i: H x W, j: H x W
    dirs = np.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -np.ones_like(i)], -1)  # dirs: H x W x 3
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                    -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))  # H x W x 3
    return rays_o, rays_d


def get_rays_by_coord_np(H, W, focal, c2w, coords):
    i, j = (coords[:, 0] - W * 0.5) / focal, -(coords[:, 1] - H * 0.5) / focal
    dirs = np.stack([i, j, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False, only_object=False, threshold=None,
                harsh_bg_remove=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)  # todo change relu to trunc_exp

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    logits = raw[..., 4]
    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    if only_object:
        if threshold is None:
            # alpha = raw2alpha(raw[..., 3] + noise, dists) * (1 - (torch.sigmoid(logits) > 0.3).float())  # [N_rays, N_samples]
            alpha = raw2alpha(raw[..., 3] + noise, dists) * (1 - torch.sigmoid(logits))  # [N_rays, N_samples]
        else:
            alpha = raw2alpha(raw[..., 3] + noise, dists) * (1 - torch.sigmoid(logits))  # [N_rays, N_samples]
            alpha[alpha > threshold] = 0  # todo check here

        if threshold is not None:
            for _ in range(5):
                alpha_right = torch.hstack([torch.zeros((alpha.shape[0], 1)), alpha[:, :-1]])
                alpha_left = torch.hstack([alpha[:, 1:], torch.zeros((alpha.shape[0], 1))])
                alpha = (alpha_right + alpha + alpha_left) / 3
    else:
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)
    prob_map = torch.sum(weights.detach() * logits, -1)  # todo check if detach is needed

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    if only_object and harsh_bg_remove:
        prob_map = prob_map - 10 * (1. - acc_map)

    return rgb_map, disp_map, acc_map, weights, depth_map, prob_map, logits


def sample_sigma(rays_o, rays_d, viewdirs, network, z_vals, network_query):
    # N_rays = rays_o.shape[0]
    # N_samples = len(z_vals)
    # z_vals = z_vals.expand([N_rays, N_samples])

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
    raw = network_query(pts, viewdirs, network)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    sigma = F.relu(raw[..., 3])

    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d)

    return rgb, sigma, depth_map


def visualize_sigma(sigma, z_vals, filename):
    plt.plot(z_vals, sigma)
    plt.xlabel('z_vals')
    plt.ylabel('sigma')
    plt.savefig(filename)
    return


def object_selection(images, img_idx):
    fig = plt.figure(figsize=(20, 30))
    img = copy.deepcopy(images[img_idx])
    labels = np.ones((img.shape[0], img.shape[1])) * (-1)

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        iy, ix = int(event.xdata), int(event.ydata)
        if ix < 0 or iy < 0:
            return

        if event.button == 1:
            img[ix][iy] = torch.tensor([1.0, 1, 1])
            labels[ix][iy] = 1
        elif event.button == 3:
            img[ix][iy] = torch.tensor([1.0, 0, 1])
            labels[ix][iy] = 0

        imgplot.set_data(img)
        plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    imgplot = plt.imshow(img)
    plt.show(block=True)
    return labels


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, preprocess = clip.load("ViT-B/32", device=device)


def img_txt_similarity(img, txt):
    image = F.interpolate(img.permute(2, 0, 1).unsqueeze(0), size=224).to(device)
    normalizer = torchvision.transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711)
    )
    image = normalizer(image)
    text = clip.tokenize([txt]).to(device)
    logits_per_image, logits_per_text = clip_model(image, text)
    return logits_per_image[0][0]


def bg_remover(frame):
    # Parameters
    blur = 21
    canny_low = 15
    canny_high = 150
    min_area = 0.02
    max_area = 0.95
    dilate_iter = 10
    erode_iter = 10
    mask_color = (1.0, 1.0, 1.0)

    # Convert image to grayscale
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Canny Edge Dection
    edges = cv2.Canny(image_gray, canny_low, canny_high)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    # get the contours and their areas
    contour_info = [(c, cv2.contourArea(c),) for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]
    # Get the area of the image as a comparison
    image_area = frame.shape[0] * frame.shape[1]
    # calculate max and min areas in terms of pixels
    max_area = max_area * image_area
    min_area = min_area * image_area
    # Set up mask with a matrix of 0's
    mask = np.zeros(edges.shape, dtype=np.uint8)
    # Go through and find relevant contours and apply to mask
    for contour in contour_info:
        # Instead of worrying about all the smaller contours, if the area is smaller than the min, the loop will break
        if contour[1] > min_area and contour[1] < max_area:
            # Add contour to mask
            mask = cv2.fillConvexPoly(mask, contour[0], (255))
    # use dilate, erode, and blur to smooth out the mask
    mask = cv2.dilate(mask, None, iterations=dilate_iter)
    mask = cv2.erode(mask, None, iterations=erode_iter)
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    # Ensures data types match up
    mask_stack = mask.astype('float32') / 255.0
    frame = frame.astype('float32') / 255.0
    # Blend the image and the mask
    mask_stack = np.expand_dims(mask_stack, axis=-1).repeat(3, axis=-1)
    masked = (mask_stack * frame) + ((1 - mask_stack) * mask_color)
    masked = (masked * 255).astype('uint8')
    return masked


def object_selection(images, img_idx):
    fig = plt.figure(figsize=(20, 30))
    img = copy.deepcopy(images[img_idx])
    labels = np.ones((img.shape[0], img.shape[1])) * (-1)

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        iy, ix = int(event.xdata), int(event.ydata)
        if ix < 0 or iy < 0:
            return

        if event.button == 1:
            img[ix][iy] = torch.tensor([1.0, 1, 1])
            labels[ix][iy] = 1
        elif event.button == 3:
            img[ix][iy] = torch.tensor([1.0, 0, 1])
            labels[ix][iy] = 0

        imgplot.set_data(img)
        plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    imgplot = plt.imshow(img)
    plt.show(block=True)
    return labels


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, preprocess = clip.load("ViT-B/32", device=device)


def img_txt_similarity(img, txt):
    image = F.interpolate(img.permute(2, 0, 1).unsqueeze(0), size=224).to(device)
    normalizer = torchvision.transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711)
    )
    image = normalizer(image)
    text = clip.tokenize([txt]).to(device)
    logits_per_image, logits_per_text = clip_model(image, text)
    return logits_per_image[0][0]


def bg_remover(frame):
    # Parameters
    blur = 21
    canny_low = 15
    canny_high = 150
    min_area = 0.02
    max_area = 0.95
    dilate_iter = 10
    erode_iter = 10
    mask_color = (1.0, 1.0, 1.0)

    # Convert image to grayscale
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Canny Edge Dection
    edges = cv2.Canny(image_gray, canny_low, canny_high)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    # get the contours and their areas
    contour_info = [(c, cv2.contourArea(c),) for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]
    # Get the area of the image as a comparison
    image_area = frame.shape[0] * frame.shape[1]
    # calculate max and min areas in terms of pixels
    max_area = max_area * image_area
    min_area = min_area * image_area
    # Set up mask with a matrix of 0's
    mask = np.zeros(edges.shape, dtype=np.uint8)
    # Go through and find relevant contours and apply to mask
    for contour in contour_info:
        # Instead of worrying about all the smaller contours, if the area is smaller than the min, the loop will break
        if contour[1] > min_area and contour[1] < max_area:
            # Add contour to mask
            mask = cv2.fillConvexPoly(mask, contour[0], (255))
    # use dilate, erode, and blur to smooth out the mask
    mask = cv2.dilate(mask, None, iterations=dilate_iter)
    mask = cv2.erode(mask, None, iterations=erode_iter)
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    # Ensures data types match up
    mask_stack = mask.astype('float32') / 255.0
    frame = frame.astype('float32') / 255.0
    # Blend the image and the mask
    mask_stack = np.expand_dims(mask_stack, axis=-1).repeat(3, axis=-1)
    masked = (mask_stack * frame) + ((1 - mask_stack) * mask_color)
    masked = (masked * 255).astype('uint8')
    return masked


def object_selection(images, img_idx):
    fig = plt.figure(figsize=(20, 30))
    img = copy.deepcopy(images[img_idx])
    labels = np.ones((img.shape[0], img.shape[1])) * (-1)

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        iy, ix = int(event.xdata), int(event.ydata)
        if ix < 0 or iy < 0:
            return

        if event.button == 1:
            img[ix][iy] = torch.tensor([1.0, 1, 1])
            labels[ix][iy] = 1
        elif event.button == 3:
            img[ix][iy] = torch.tensor([1.0, 0, 1])
            labels[ix][iy] = 0

        imgplot.set_data(img)
        plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    imgplot = plt.imshow(img)
    plt.show(block=True)
    return labels


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, preprocess = clip.load("ViT-B/32", device=device)


def img_txt_similarity(img, txt):
    image = F.interpolate(img.permute(2, 0, 1).unsqueeze(0), size=224).to(device)
    normalizer = torchvision.transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711)
    )
    image = normalizer(image)
    text = clip.tokenize([txt]).to(device)
    logits_per_image, logits_per_text = clip_model(image, text)
    return logits_per_image[0][0]


def bg_remover(frame):
    # Parameters
    blur = 21
    canny_low = 15
    canny_high = 150
    min_area = 0.02
    max_area = 0.95
    dilate_iter = 10
    erode_iter = 10
    mask_color = (1.0, 1.0, 1.0)

    # Convert image to grayscale
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Canny Edge Dection
    edges = cv2.Canny(image_gray, canny_low, canny_high)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    # get the contours and their areas
    contour_info = [(c, cv2.contourArea(c),) for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]
    # Get the area of the image as a comparison
    image_area = frame.shape[0] * frame.shape[1]
    # calculate max and min areas in terms of pixels
    max_area = max_area * image_area
    min_area = min_area * image_area
    # Set up mask with a matrix of 0's
    mask = np.zeros(edges.shape, dtype=np.uint8)
    # Go through and find relevant contours and apply to mask
    for contour in contour_info:
        # Instead of worrying about all the smaller contours, if the area is smaller than the min, the loop will break
        if contour[1] > min_area and contour[1] < max_area:
            # Add contour to mask
            mask = cv2.fillConvexPoly(mask, contour[0], (255))
    # use dilate, erode, and blur to smooth out the mask
    mask = cv2.dilate(mask, None, iterations=dilate_iter)
    mask = cv2.erode(mask, None, iterations=erode_iter)
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    # Ensures data types match up
    mask_stack = mask.astype('float32') / 255.0
    frame = frame.astype('float32') / 255.0
    # Blend the image and the mask
    mask_stack = np.expand_dims(mask_stack, axis=-1).repeat(3, axis=-1)
    masked = (mask_stack * frame) + ((1 - mask_stack) * mask_color)
    masked = (masked * 255).astype('uint8')
    return masked


def object_selection(images, img_idx):
    fig = plt.figure(figsize=(20, 30))
    img = copy.deepcopy(images[img_idx])
    labels = np.ones((img.shape[0], img.shape[1])) * (-1)

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        iy, ix = int(event.xdata), int(event.ydata)
        if ix < 0 or iy < 0:
            return

        if event.button == 1:
            img[ix][iy] = torch.tensor([1.0, 1, 1])
            labels[ix][iy] = 1
        elif event.button == 3:
            img[ix][iy] = torch.tensor([1.0, 0, 1])
            labels[ix][iy] = 0

        imgplot.set_data(img)
        plt.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    imgplot = plt.imshow(img)
    plt.show(block=True)
    return labels


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model, preprocess = clip.load("ViT-B/32", device=device)


def img_txt_similarity(img, txt):
    image = F.interpolate(img.permute(2, 0, 1).unsqueeze(0), size=224).to(device)
    normalizer = torchvision.transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711)
    )
    image = normalizer(image)
    text = clip.tokenize([txt]).to(device)
    logits_per_image, logits_per_text = clip_model(image, text)
    return logits_per_image[0][0]


def bg_remover(frame):
    # Parameters
    blur = 21
    canny_low = 15
    canny_high = 150
    min_area = 0.02
    max_area = 0.95
    dilate_iter = 10
    erode_iter = 10
    mask_color = (1.0, 1.0, 1.0)

    # Convert image to grayscale
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Canny Edge Dection
    edges = cv2.Canny(image_gray, canny_low, canny_high)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    # get the contours and their areas
    contour_info = [(c, cv2.contourArea(c),) for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]
    # Get the area of the image as a comparison
    image_area = frame.shape[0] * frame.shape[1]
    # calculate max and min areas in terms of pixels
    max_area = max_area * image_area
    min_area = min_area * image_area
    # Set up mask with a matrix of 0's
    mask = np.zeros(edges.shape, dtype=np.uint8)
    # Go through and find relevant contours and apply to mask
    for contour in contour_info:
        # Instead of worrying about all the smaller contours, if the area is smaller than the min, the loop will break
        if contour[1] > min_area and contour[1] < max_area:
            # Add contour to mask
            mask = cv2.fillConvexPoly(mask, contour[0], (255))
    # use dilate, erode, and blur to smooth out the mask
    mask = cv2.dilate(mask, None, iterations=dilate_iter)
    mask = cv2.erode(mask, None, iterations=erode_iter)
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    # Ensures data types match up
    mask_stack = mask.astype('float32') / 255.0
    frame = frame.astype('float32') / 255.0
    # Blend the image and the mask
    mask_stack = np.expand_dims(mask_stack, axis=-1).repeat(3, axis=-1)
    masked = (mask_stack * frame) + ((1 - mask_stack) * mask_color)
    masked = (masked * 255).astype('uint8')
    return masked
