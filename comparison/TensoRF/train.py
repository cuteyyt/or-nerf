import datetime

import lpips
from torch.utils.tensorboard import SummaryWriter

from dataLoader import dataset_dict
from models.tensoRF import TensorVMSplit
from opt import config_parser
from renderer import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
perceptual_loss = lpips.LPIPS(net='vgg').to(device)
for param in perceptual_loss.parameters():
    param.requires_grad = False

renderer = OctreeRender_trilinear_fast


class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr + self.batch]


@torch.no_grad()
def export_mesh(args):
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha, _ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply', bbox=tensorf.aabb.cpu(), level=0.005)


@torch.no_grad()
def render_test(args):
    print("-----------START RENDERING-----------")
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    # logfolder = os.path.dirname(args.ckpt)
    logfolder = args.basedir
    print(logfolder)
    if args.render_train:
        print("-----------render train-----------")
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset, tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        print("-----------render test-----------")
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        evaluation(test_dataset, tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_test_all/',
                   N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)

    if args.render_path:
        print("-----------render path-----------")
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset, tensorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
                        N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)

    if args.render_all:
        print("-----------render all-----------")
        os.makedirs(f'{logfolder}/{args.expname}/imgs_all', exist_ok=True)
        all_dataset = dataset(args.datadir, split='all', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(all_dataset, tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_all/',
                                N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)
    
    if args.render_gt:
        print("-----------render gt-----------")
        os.makedirs(f'{logfolder}/{args.expname}/imgs_gt', exist_ok=True)
        all_dataset = dataset(args.datadir, split='all', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(all_dataset, tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_gt/',
                                N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)


def cal_box(mask):
    # (h, w, 1)
    # print(np.nonzero(mask[:, :, 0]))
    x_coords, y_coords = np.nonzero(mask[:, :, 0]).split([1, 1], dim=1)
    x_coords = torch.squeeze(x_coords)
    y_coords = torch.squeeze(y_coords)
    res = np.asarray([x_coords.min, y_coords.min, x_coords.max, y_coords.max])
    # print(x_coords)
    return res


def reconstruction(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False,
                            is_depth=args.depth, is_mask=args.lpips)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True,
                           is_depth=args.depth, is_mask=args.lpips)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh
    no_bb = args.no_bb

    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
    aabb = train_dataset.scene_bbox.to(device)
    # print(aabb)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device': device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = TensorVMSplit(aabb, reso_cur, device,
                                density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh,
                                app_dim=args.data_dim_color, near_far=near_far,
                                shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre,
                                density_shift=args.density_shift, distance_scale=args.distance_scale,
                                pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe,
                                featureC=args.featureC, step_ratio=args.step_ratio,
                                fea2denseAct=args.fea2denseAct)

    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio ** (1 / args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)

    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    # linear in logrithmic space
    N_voxel_list = (torch.round(torch.exp(
        torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list) + 1))).long()).tolist()[
                   1:]

    torch.cuda.empty_cache()
    PSNRs, PSNRs_test = [], [0]

    allrays, allrgbs, alldepth = train_dataset.all_rays, train_dataset.all_rgbs, train_dataset.all_depth
    if not args.ndc_ray:
        allrays, allrgbs, alldepth = tensorf.filtering_rays(allrays, allrgbs, alldepth, bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)

    Depth_loss_weight = args.Depth_loss_weight
    Lpips_loss_weight = args.Lpips_loss_weight
    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)
    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")

    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    cnt = 0
    cnt_1 = 0
    for iteration in pbar:

        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device),
        if len(alldepth):
            depth_train = alldepth[ray_idx].to(device)

        # rgb_map, alphas_map, depth_map, weights, uncertainty
        # print("rays train: ", rays_train.shape)
        rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(rays_train, tensorf, chunk=args.batch_size,
                                                                        N_samples=nSamples, white_bg=white_bg,
                                                                        ndc_ray=ndc_ray, device=device, is_train=True)

        loss = torch.mean((rgb_map - rgb_train) ** 2)
        # print("rgb loss---------> ", loss)

        # loss
        total_loss = loss
        if Lpips_loss_weight > 0:
            rgbs, targets = list(), list()

            img_list, _H, _W = train_dataset.get_same_rays()
            images, masks, poses = train_dataset.images, train_dataset.masks, train_dataset.poses
            focal, directions = train_dataset.focal, train_dataset.directions

            for i in img_list:
                c2w = torch.FloatTensor(poses[i])
                image = images[i]
                mask = masks[i]
                box = train_dataset.cal_box(mask)

                
                img_shape = np.array([_H, _W])
                if args.datadir.endswith("qq16") or args.datadir.endswith("qq6"):
                    box_shape = np.array([box[3] - box[1], box[2] - box[0]])
                    patch_shape = box_shape // 4
                    patch_shape = np.array([np.clip(patch_shape[0], 32, 64), np.clip(patch_shape[1], 32, 64)])
                else:
                    patch_shape = img_shape // 16

                rays_o, rays_d = get_rays(directions, c2w)  # both (h*w, 3)
                rays_o, rays_d = ndc_rays_blender(_H, _W, focal[0], 1.0, rays_o.to(device), rays_d.to(device))
                rays_o = rays_o.reshape(_H, _W, 3)
                rays_d = rays_d.reshape(_H, _W, 3)

                try:
                    cnt += 1
                    random_point_x = np.random.randint(box[0], max(box[2] - patch_shape[1], box[0]))
                    random_point_y = np.random.randint(box[1], max(box[3] - patch_shape[0], box[1]))
                except:
                    cnt_1 += 1
                    continue

                patch_indices_x = torch.linspace(
                    random_point_x, random_point_x + patch_shape[1], patch_shape[1], dtype=torch.int32)
                patch_indices_y = torch.linspace(
                    random_point_y, random_point_y + patch_shape[0], patch_shape[0], dtype=torch.int32)
                select_coords = torch.tensor(
                    [[[patch_indices_x[m], patch_indices_y[n]] for n in range(patch_shape[0])] for m in
                     range(patch_shape[1])])
                select_coords = select_coords.reshape(-1, 2).long()

                target_patch = image[select_coords[:, 0], select_coords[:, 1]]
                target_patch = torch.stack(
                    [target_patch[k * patch_shape[1]:(k + 1) * patch_shape[1], :] for k in range(patch_shape[0])])
                target_patch = target_patch.permute(2, 0, 1).unsqueeze(0)
                targets.append(target_patch)

                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays = torch.cat([rays_o, rays_d], 1)

                
                rgb_p, alphas_map, depth_p, weights, uncertainty = renderer(rays, tensorf, chunk=args.batch_size,
                                                                            N_samples=nSamples, white_bg=white_bg,
                                                                            ndc_ray=ndc_ray, device=device,
                                                                            is_train=True)
                rgb_p = torch.stack(
                    [rgb_p[k * patch_shape[1]:(k + 1) * patch_shape[1], :] for k in range(patch_shape[0])])
                rgb_p = rgb_p.permute(2, 0, 1).unsqueeze(0)
                rgbs.append(rgb_p)

            targets = torch.cat(targets).to(device)
            rgbs = torch.cat(rgbs).to(device)

            total_loss += Lpips_loss_weight * perceptual_loss(rgbs, targets).mean()

        if Depth_loss_weight > 0:
            # depth_vis, _ = visualize_depth_numpy(depth_map.cpu().numpy(), train_dataset.near_far)
            # depth_vis = torch.tensor(np.squeeze(depth_vis)).to(device)
            depth_loss = torch.mean((depth_map[..., None] - depth_train) ** 2)
            total_loss += Depth_loss_weight * depth_loss
            summary_writer.add_scalar('train/depth', depth_loss.detach().item(), global_step=iteration)
        if Ortho_reg_weight > 0:
            loss_reg = tensorf.vector_comp_diffs()
            total_loss += Ortho_reg_weight * loss_reg
            summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)
        if L1_reg_weight > 0:
            loss_reg_L1 = tensorf.density_L1()
            total_loss += L1_reg_weight * loss_reg_L1
            summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

        if TV_weight_density > 0:
            TV_weight_density *= lr_factor
            loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
        if TV_weight_app > 0:
            TV_weight_app *= lr_factor
            loss_tv = tensorf.TV_loss_app(tvreg) * TV_weight_app
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss = loss.detach().item()

        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/mse', loss, global_step=iteration)

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' mse = {loss:.6f}'
            )
            PSNRs = []

        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis != 0:
            PSNRs_test = evaluation(test_dataset, tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                    prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg=white_bg, ndc_ray=ndc_ray,
                                    compute_extra_metrics=False)
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)

        if not no_bb:
            # print("update bounding box")
            if iteration in update_AlphaMask_list:

                if reso_cur[0] * reso_cur[1] * reso_cur[2] < 256 ** 3:  # update volume resolution
                    reso_mask = reso_cur
                new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
                if iteration == update_AlphaMask_list[0]:
                    tensorf.shrink(new_aabb)
                    # tensorVM.alphaMask = None
                    L1_reg_weight = args.L1_weight_rest
                    print("continuing L1_reg_weight", L1_reg_weight)

                if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                    # filter rays outside the bbox
                    allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs)
                    trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size)

        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))
            tensorf.upsample_volume_grid(reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1  # 0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensorf.get_optparam_groups(args.lr_init * lr_scale, args.lr_basis * lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    tensorf.save(f'{logfolder}/{args.expname}.th')

    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset, tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset, tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print('========>', c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset, tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                        N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)

    if args.render_all:
        os.makedirs(f'{logfolder}/imgs_all', exist_ok=True)
        all_dataset = dataset(args.datadir, split='all', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(all_dataset, tensorf, args, renderer, f'{logfolder}/imgs_all/',
                                N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)
    
    print("try and not try: ", cnt, cnt_1)
    np.savetxt(f'{logfolder}/trynum.txt', np.asarray([cnt, cnt_1, cnt/(cnt+cnt_1+1)]))


if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)

    if args.export_mesh:
        export_mesh(args)

    if args.render_only and (args.render_test or args.render_path or args.render_all):
        render_test(args)
    else:
        reconstruction(args)
