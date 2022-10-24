import time

import matplotlib.pyplot as plt
from pandas import DataFrame

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
from pytorch3d.renderer.cameras import camera_position_from_spherical_angles

sys.path.append("..")
from ops import *
from torch.utils.tensorboard import SummaryWriter
import math
from torchvision import transforms
from torchvision import utils as vutils

# rendering components
PLOT_SAMPLE_NBS = [240, 4, 150, 47, 110]
VISUALIZER_PRE = False
VIS_REAL_PATH_POINT = './output/gt/'
if VISUALIZER_PRE == True:
    if not os.path.isdir(VIS_REAL_PATH_POINT):
        os.makedirs(VIS_REAL_PATH_POINT)

class ModelNetTrainer_mvt_color(object):
    def __init__(self, models_bag, train_loader, val_loader, val_set, loss_fn,
                 model_name, weights_dir, num_views=12, setup=None, classes=[], device = "cuda"):
        self.models_bag = models_bag
        self.model = self.models_bag["mvnetwork"]
        self.optimizer = self.models_bag["optimizer"]

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_set = val_set
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.weights_dir = weights_dir
        self.num_views = num_views
        self.setup = setup
        self.classes = classes
        self.device = device
        if self.setup["log_metrics"]:
            self.writer = SummaryWriter(setup["logs_dir"])
            # self.writer = SummaryWriter(self.weights_dir)

    def Normalize(self, x, ignore_normalize=False):
        if ignore_normalize:
            return x
        else:
            mean = torch.Tensor([0.456, 0.456, 0.456]).unsqueeze(
                0).unsqueeze(2).unsqueeze(3).repeat(x.shape[0], 1, x.shape[2], x.shape[3])
            std = torch.Tensor([0.225, 0.225, 0.225]).unsqueeze(
                0).unsqueeze(2).unsqueeze(3).repeat(x.shape[0], 1, x.shape[2], x.shape[3])
            return (x - mean.cuda()) / std.cuda()

    def train(self, n_epochs):
        # best_acc = 0
        i_acc = 0
        self.model.train()
        saveables = {}
        for epoch in range(self.setup["start_epoch"],n_epochs):
            print("epoch == " + str(epoch))
            start_time = time.time()
            self.setup["c_epoch"] = epoch
            if self.model_name == 'view_gcn':
                if epoch == 1:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                if epoch > 1:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.5 * (1 + math.cos(epoch * math.pi / 15))
            else:
                if epoch > 0 and (epoch + 1) % 10 == 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.5

            rand_idx = np.random.permutation(int(len(self.train_loader.dataset)))

            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            if self.setup["log_metrics"]:
                self.writer.add_scalar('params/lr', lr, epoch)
            # train one epoch
            out_data = None
            in_data = None
            train_size = len(self.train_loader)
            for i, (targets, meshes, points) in enumerate(self.train_loader):

                c_batch_size = targets.shape[0]

                targets = targets.to(self.device).long()
                points = points.to(self.device)
                targets = Variable(targets)

                azim, elev, dist = self.models_bag["mvtn"](
                    points, c_batch_size=c_batch_size)

                color = None
                if self.setup["object_color"] == "custom":
                    color = self.models_bag["color"](points.permute(0, 2, 1))

                rendered_images, _ = self.models_bag["mvrenderer"](
                    meshes, points, azim=azim, elev=elev, dist=dist, color=color)


                N, V, C, H, W = rendered_images.size()
                rendered_images = regualarize_rendered_views(rendered_images, self.setup["view_reg"],
                                                             self.setup["augment_training"], self.setup["crop_ratio"])

                N, V, C, H, W = rendered_images.size()
                rendered_images = self.Normalize(rendered_images.contiguous().view(-1, C, H, W),
                                                 ignore_normalize=self.setup["ignore_normalize"])
                if VISUALIZER_PRE == True:
                    for k in range(rendered_images.size(0)):
                        img = rendered_images[k, :, :, :]
                        vutils.save_image(img, VIS_REAL_PATH_POINT + '{}.jpg'.format(str(k)),ormalize=True)

                if self.model_name == 'svcnn':
                    targets = targets.repeat_interleave(V)

                if self.model_name == 'view-gcn' and epoch == 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr * ((i + 1) / (len(rand_idx) // 20))

                if self.num_views == 30:
                    targets_ = targets.unsqueeze(1).repeat(1, 4 * (10 + 12)).view(-1)
                elif self.num_views == 20:
                    targets_ = targets.unsqueeze(1).repeat(1, 4 * (10 + 5)).view(-1)            # 1200
                elif self.num_views == 12:
                    targets_ = targets.unsqueeze(1).repeat(1, 4 * (10 + 0)).view(-1)            # 800
                elif self.num_views == 6:
                    targets_ = targets.unsqueeze(1).repeat(1, 4 * (8  + 0)).view(-1)            # 640

                self.optimizer.zero_grad()
                if self.setup["object_color"] == "custom" and (self.model_name == 'svcnn' or
                                                               (self.model_name == 'view-gcn' and not self.setup["no_joint_train"])):
                    self.models_bag["color_optimizer"].zero_grad()
                if self.setup["is_learning_views"]:
                    self.models_bag["mvtn_optimizer"].zero_grad()

                if self.model_name == 'view-gcn':
                    self.model.vertices = unbatch_tensor(
                        camera_position_from_spherical_angles(distance=batch_tensor(
                            dist.T, dim=1, squeeze=True), elevation=batch_tensor(elev.T, dim=1, squeeze=True),
                            azimuth=batch_tensor(azim.T, dim=1, squeeze=True)), batch_size=self.setup["nb_views"],
                        dim=1, unsqueeze=True).transpose(0, 1).to(targets.device)  # torch.Size([30, 12, 3])

                    out_data, F_score, F_score2 = self.model(rendered_images)
                    out_data_ = torch.cat(
                        (F_score, F_score2), 1).view(-1, len(self.classes))
                    loss = self.loss_fn(out_data, targets) + self.loss_fn(out_data_, targets_)
                else:
                    out_data = self.model(rendered_images)
                    loss = self.loss_fn(out_data, targets)
                if self.setup["log_metrics"]:
                    self.writer.add_scalar('train/train_loss', loss, i_acc + i + 1)

                pred = torch.max(out_data, 1)[1]
                results = pred == targets
                correct_points = torch.sum(results.long())

                if (i + 1) % self.setup["print_freq"] == 0:
                    print("\tIter [%d/%d] Loss: %.4f" % (i + 1, train_size, loss))
                acc = correct_points.float() / results.size()[0]
                if self.setup["log_metrics"]:
                    self.writer.add_scalar('train/train_overall_acc', acc, i_acc + i + 1)

                loss.backward()

                self.optimizer.step()
                if self.setup["object_color"] == "custom" and (self.model_name == 'svcnn' or
                                                               (self.model_name == 'view-gcn' and not self.setup["no_joint_train"])):
                    self.models_bag["color_optimizer"].step()
                if self.setup["is_learning_views"]:
                    self.models_bag["mvtn_optimizer"].step()
                    if self.setup["clip_grads"]:
                        clip_grads_(self.models_bag["mvtn"].parameters(
                        ), self.setup["mvtn_clip_grads_value"])
                    if self.setup["log_metrics"]:
                        step = get_current_step(self.models_bag["mvtn_optimizer"])
                        self.writer.add_scalar('Zoom/loss', loss.item(), step)
                        self.writer.add_scalar(
                            'Zoom/MVT_vals', list(self.models_bag["mvtn"].parameters())[0].data[0, 0].item(), step)
                        self.writer.add_scalar('Zoom/MVT_grads', np.sum(np.array([np.sum(x.grad.cpu(
                        ).numpy() ** 2) for x in self.models_bag["mvtn"].parameters()])), step)

            i_acc += i
            # evaluation
            with torch.no_grad():
                loss, val_overall_acc, val_mean_class_acc, views_record = self.update_validation_accuracy(epoch)
            if self.setup["log_metrics"]:
                self.writer.add_scalar('val/val_mean_class_acc', val_mean_class_acc, epoch + 1)
                self.writer.add_scalar('val/val_overall_acc', val_overall_acc, epoch + 1)
                self.writer.add_scalar('val/val_loss', loss, epoch + 1)

            saveables = {'epoch': epoch + 1,
                         'state_dict': self.model.state_dict(),
                         "mvtn": self.models_bag["mvtn"].state_dict(),
                         "color": self.models_bag["color"].state_dict(),
                         'acc': val_overall_acc,
                         'best_acc': self.setup["best_acc"],
                         'optimizer': self.models_bag["optimizer"].state_dict(),
                         'mvtn_optimizer': None if not self.setup["is_learning_views"] else self.models_bag[
                             "mvtn_optimizer"].state_dict(),
                         'color_optimizer': None if not self.setup["object_color"] == "custom" else self.models_bag[
                             "color_optimizer"].state_dict(),
                         }
            if self.setup["save_all"]:
                print("saving results ..")
                save_checkpoint(saveables, self.setup, views_record,
                                os.path.join(self.weights_dir, self.setup["exp_id"] + "_checkpoint.pt"))

            # Save mvnetwork
            # save best model
            if val_overall_acc > self.setup["best_acc"]:
                self.setup["best_acc"] = val_overall_acc
                self.setup["best_cls_avg_acc"] = val_mean_class_acc
                saveables["best_acc"] = val_overall_acc
                self.setup["best_loss"] = loss

                save_checkpoint(saveables, self.setup, views_record, os.path.join(
                    self.weights_dir, self.setup["exp_id"] + "_checkpoint.pt"))
            print('best_acc', self.setup["best_acc"])

            if (epoch + 1) % self.setup["plot_freq"] == 0:
                self.visualize_views(epoch, PLOT_SAMPLE_NBS)

            end_time = time.time()
            print("耗时: {:.2f}秒".format(end_time - start_time))

        if self.setup["log_metrics"]:
            self.writer.add_hparams(self.setup, {"hparams/best_acc": self.setup["best_acc"],
                                                 "hparams/best_cls_avg_acc": self.setup["best_cls_avg_acc"],
                                                 "hparams/retr_map": 0,
                                                 "hparams/retr_pn": 0,
                                                 "hparams/retr_rn": 0,
                                                 "hparams/retr_fn": 0,
                                                 })
        save_checkpoint(saveables, self.setup, views_record, os.path.join(self.weights_dir, self.setup["exp_id"] + "_checkpoint_final.pt"))

    def visualize_views(self, epoch, object_nbs):
        self.model.eval()
        self.models_bag["mvtn"].eval()
        self.models_bag["color"].eval()
        self.models_bag["mvrenderer"].eval()
        # self.models_bag["feature_extractor"].eval()

        for indx, ii in enumerate(object_nbs):
            (targets, meshes, points) = self.val_set[ii]
            c_batch_size = 1
            targets = torch.tensor(targets).to(self.device)
            if "Scan" in self.setup["data_dir"] or "ShapeNetCore" in self.setup["data_dir"]:
                points = points.to(self.device)
            else:
                points = torch.from_numpy(points).to(self.device)
            cameras_root_folder = os.path.join(
                self.setup["cameras_dir"], str(indx))
            check_folder(cameras_root_folder)
            renderings_root_folder = os.path.join(
                self.setup["renderings_dir"], str(indx))
            check_folder(renderings_root_folder)
            cameras_path = os.path.join(
                cameras_root_folder, "MV_cameras_{}.jpg".format(epoch))
            images_path = os.path.join(
                renderings_root_folder, "MV_renderings_{}.jpg".format(epoch))
            # points = torch.from_numpy(points)
            if not self.setup["return_points_saved"] and not self.setup["return_points_sampled"]:
                points = torch.from_numpy(points)

            azim, elev, dist = self.models_bag["mvtn"](
                points[None, ...], c_batch_size=c_batch_size)

            color = None
            if self.setup["object_color"] == "custom":
                color = self.models_bag["color"](points[None, ...].permute(0, 2, 1))

            self.models_bag["mvrenderer"].render_and_save(
                [meshes], points[None, ...], azim=azim, elev=elev, dist=dist, images_path=images_path,
                cameras_path=cameras_path, color=color)


        self.model.train()
        self.models_bag["mvtn"].train()
        self.models_bag["color"].train()
        self.models_bag["mvrenderer"].train()
        if self.model_name == 'view-gcn' and self.setup["no_joint_train"]:
            self.models_bag["color"].eval()

    def update_validation_accuracy(self, epoch, path=None):
        all_correct_points = 0
        all_points = 0
        ranked_losses = []
        wrong_class = np.zeros(len(self.classes))
        samples_class = np.zeros(len(self.classes))
        all_loss = 0
        self.model.eval()
        self.models_bag["mvtn"].eval()
        self.models_bag["color"].eval()
        self.models_bag["mvrenderer"].eval()

        views_record = ListDict(
            ["azim", "elev", "dist", "label", "view_nb", "exp_id"])

        for i, (targets, meshes, points) in enumerate(tqdm(self.val_loader)):
            with torch.no_grad():
                c_batch_size = targets.shape[0]
                targets = targets.to(self.device).long()
                points = points.to(self.device)
                targets = Variable(targets)

                azim, elev, dist = self.models_bag["mvtn"](
                    points, c_batch_size=c_batch_size)

                color = None
                if self.setup["object_color"] == "custom":
                    color = self.models_bag["color"](points.permute(0, 2, 1))

                rendered_images, _ = self.models_bag["mvrenderer"](
                    meshes, points, azim=azim, elev=elev, dist=dist, color=color)

                N, V, C, H, W = rendered_images.size()
                rendered_images = self.Normalize(rendered_images.contiguous(
                ).view(-1, C, H, W), ignore_normalize=self.setup["ignore_normalize"])

                if VISUALIZER_PRE == True:
                    for k in range(rendered_images.size(0)):
                        img = rendered_images[k, :, :, :]
                        vutils.save_image(img,VIS_REAL_PATH_POINT + '{}.jpg'.format(str(k)),ormalize=True)

                if self.model_name == 'svcnn':
                    targets = targets.repeat_interleave(V)

                if self.model_name == 'view-gcn':
                    self.model.vertices = unbatch_tensor(
                        camera_position_from_spherical_angles(distance=batch_tensor(
                            dist.T, dim=1, squeeze=True), elevation=batch_tensor(elev.T, dim=1, squeeze=True),
                            azimuth=batch_tensor(azim.T, dim=1, squeeze=True)), batch_size=self.setup["nb_views"],
                        dim=1, unsqueeze=True).transpose(0, 1).to(targets.device)  # torch.Size([30, 12, 3])
                    out_data, F1, F2 = self.model(rendered_images)
                else:
                    aa = nn.Sequential(*list(self.model.net.children())[:-1])(rendered_images)
                    out_data = self.model(rendered_images)

                pred = torch.max(out_data, 1)[1]
                all_loss += self.loss_fn(out_data, targets).cpu().data.numpy()
                results = pred == targets
                ################################
                wrong_predictions = torch.nonzero((pred != targets).to(torch.uint8)) + i * targets.shape[0]
                ranked_losses.extend(wrong_predictions.cpu().data.numpy().tolist())
                ###################################

                for i in range(results.size()[0]):
                    if not bool(results[i].cpu().data.numpy()):
                        wrong_class[targets.cpu().data.numpy().astype('int')[i]] += 1
                    samples_class[targets.cpu().data.numpy().astype('int')[i]] += 1
                correct_points = torch.sum(results.long())

                all_correct_points += correct_points
                all_points += results.size()[0]
                if self.model_name == 'svcnn':
                    c_views = ListDict({"azim": azim.cpu().numpy().reshape(-1).tolist(),
                                        "elev": elev.cpu().numpy().reshape(-1).tolist(),
                                        "dist": dist.cpu().numpy().reshape(-1).tolist(),
                                        "label": targets.cpu().numpy().tolist(),
                                        "view_nb": int(targets.cpu().numpy().shape[0]) * [0],
                                        "exp_id": int(targets.cpu().numpy().shape[0]) * [self.setup["exp_id"]]})
                else:
                    c_views = ListDict({"azim": azim.cpu().numpy().reshape(-1).tolist(),
                                        "elev": elev.cpu().numpy().reshape(-1).tolist(),
                                        "dist": dist.cpu().numpy().reshape(-1).tolist(),
                                        "label": np.repeat(targets.cpu().numpy(), self.setup["nb_views"]).tolist(),
                                        "view_nb": int(targets.cpu().numpy().shape[0]) * list(
                                            range(self.setup["nb_views"])),
                                        "exp_id": int(targets.cpu().numpy().shape[0]) * int(self.setup["nb_views"]) * [
                                            self.setup["exp_id"]]})
                views_record.extend(c_views)

        print('Total # of test models: ', all_points)
        class_acc = (samples_class - wrong_class) / samples_class
        val_mean_class_acc = np.mean(class_acc)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)

        print('val mean class acc. : ', val_mean_class_acc)
        print('val overall acc. : ', val_overall_acc)
        print('val loss : ', loss)
        print(class_acc)
        self.model.train()
        self.models_bag["mvtn"].train()
        self.models_bag["color"].train()
        self.models_bag["mvrenderer"].train()
        if self.model_name == 'view-gcn' and self.setup["no_joint_train"]:
            self.models_bag["color"].eval()

        return loss.item(), val_overall_acc.item(), val_mean_class_acc.item(), views_record

    def update_validation_accuracy_train(self, epoch, path=None):
        all_correct_points = 0
        all_points = 0
        ranked_losses = []
        wrong_class = np.zeros(len(self.classes))
        samples_class = np.zeros(len(self.classes))
        all_loss = 0
        self.model.eval()
        self.models_bag["mvtn"].eval()
        self.models_bag["color"].eval()
        self.models_bag["mvrenderer"].eval()

        views_record = ListDict(
            ["azim", "elev", "dist", "label", "view_nb", "exp_id"])

        for i, (targets, meshes, points) in enumerate(tqdm(self.train_loader)):
            with torch.no_grad():
                c_batch_size = targets.shape[0]
                targets = targets.to(self.device).long()
                points = points.to(self.device)
                targets = Variable(targets)

                azim, elev, dist = self.models_bag["mvtn"](
                    points, c_batch_size=c_batch_size)

                color = None
                if self.setup["object_color"] == "custom":
                    color = self.models_bag["color"](points.permute(0, 2, 1))

                rendered_images, _ = self.models_bag["mvrenderer"](
                    meshes, points, azim=azim, elev=elev, dist=dist, color=color)

                N, V, C, H, W = rendered_images.size()
                rendered_images = self.Normalize(rendered_images.contiguous(
                ).view(-1, C, H, W), ignore_normalize=self.setup["ignore_normalize"])

                if VISUALIZER_PRE == True:
                    for k in range(rendered_images.size(0)):
                        img = rendered_images[k, :, :, :]
                        vutils.save_image(img, VIS_REAL_PATH_POINT + '{}.jpg'.format(str(k)), ormalize=True)

                if self.model_name == 'svcnn':
                    targets = targets.repeat_interleave(V)

                if self.model_name == 'view-gcn':
                    self.model.vertices = unbatch_tensor(
                        camera_position_from_spherical_angles(distance=batch_tensor(
                            dist.T, dim=1, squeeze=True), elevation=batch_tensor(elev.T, dim=1, squeeze=True),
                            azimuth=batch_tensor(azim.T, dim=1, squeeze=True)), batch_size=self.setup["nb_views"],
                        dim=1, unsqueeze=True).transpose(0, 1).to(targets.device)  # torch.Size([30, 12, 3])
                    out_data, F1, F2 = self.model(rendered_images)
                else:
                    out_data = self.model(rendered_images)

                pred = torch.max(out_data, 1)[1]
                all_loss += self.loss_fn(out_data, targets).cpu().data.numpy()
                results = pred == targets
                ################################
                wrong_predictions = torch.nonzero((pred != targets).to(torch.uint8)) + i * targets.shape[0]
                ranked_losses.extend(wrong_predictions.cpu().data.numpy().tolist())
                ###################################

                for i in range(results.size()[0]):
                    if not bool(results[i].cpu().data.numpy()):
                        wrong_class[targets.cpu().data.numpy().astype('int')[i]] += 1
                    samples_class[targets.cpu().data.numpy().astype('int')[i]] += 1
                correct_points = torch.sum(results.long())

                all_correct_points += correct_points
                all_points += results.size()[0]
                if self.model_name == 'svcnn':
                    c_views = ListDict({"azim": azim.cpu().numpy().reshape(-1).tolist(),
                                        "elev": elev.cpu().numpy().reshape(-1).tolist(),
                                        "dist": dist.cpu().numpy().reshape(-1).tolist(),
                                        "label": targets.cpu().numpy().tolist(),
                                        "view_nb": int(targets.cpu().numpy().shape[0]) * [0],
                                        "exp_id": int(targets.cpu().numpy().shape[0]) * [self.setup["exp_id"]]})
                else:
                    c_views = ListDict({"azim": azim.cpu().numpy().reshape(-1).tolist(),
                                        "elev": elev.cpu().numpy().reshape(-1).tolist(),
                                        "dist": dist.cpu().numpy().reshape(-1).tolist(),
                                        "label": np.repeat(targets.cpu().numpy(), self.setup["nb_views"]).tolist(),
                                        "view_nb": int(targets.cpu().numpy().shape[0]) * list(
                                            range(self.setup["nb_views"])),
                                        "exp_id": int(targets.cpu().numpy().shape[0]) * int(self.setup["nb_views"]) * [
                                            self.setup["exp_id"]]})
                views_record.extend(c_views)

        print('Total # of test models: ', all_points)
        class_acc = (samples_class - wrong_class) / samples_class
        val_mean_class_acc = np.mean(class_acc)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)

        print('val mean class acc. : ', val_mean_class_acc)
        print('val overall acc. : ', val_overall_acc)
        print('val loss : ', loss)
        print(class_acc)
        self.model.train()
        self.models_bag["mvtn"].train()
        self.models_bag["color"].train()
        self.models_bag["mvrenderer"].train()
        if self.model_name == 'view-gcn' and self.setup["no_joint_train"]:
            self.models_bag["color"].eval()

        return loss.item(), val_overall_acc.item(), val_mean_class_acc.item(), views_record

    def compute_features(self, path=None):
        all_correct_points = 0
        all_points = 0
        count = 0
        wrong_class = np.zeros(len(self.classes))
        samples_class = np.zeros(len(self.classes))
        all_loss = 0
        self.model.eval()
        self.models_bag["mvtn"].eval()
        self.models_bag["mvrenderer"].eval()
        self.models_bag["color"].eval()
        feat_list = []
        target_list = []
        from tqdm import tqdm
        for i, (targets, meshes, points) in tqdm(enumerate(self.train_loader)):
            with torch.no_grad():
                c_batch_size = targets.shape[0]
                targets = targets.to(self.device).long()
                points = points.to(self.device)
                targets = Variable(targets)
                azim, elev, dist = self.models_bag["mvtn"](
                    points, c_batch_size=c_batch_size)
                color = None
                if self.setup["object_color"] == "custom":
                    color = self.models_bag["color"](points.permute(0, 2, 1))

                rendered_images, _ = self.models_bag["mvrenderer"](
                    meshes, points, azim=azim, elev=elev, dist=dist, color=color)
                N, V, C, H, W = rendered_images.size()
                rendered_images = self.Normalize(rendered_images.contiguous(
                ).view(-1, C, H, W), ignore_normalize=self.setup["ignore_normalize"])

                if self.model_name == 'svcnn':
                    targets = targets.repeat_interleave(V)

                if self.model_name == 'view-gcn':
                    self.model.vertices = unbatch_tensor(
                        camera_position_from_spherical_angles(distance=batch_tensor(
                            dist.T, dim=1, squeeze=True), elevation=batch_tensor(elev.T, dim=1, squeeze=True),
                            azimuth=batch_tensor(azim.T, dim=1, squeeze=True)), batch_size=self.setup["nb_views"],
                        dim=1, unsqueeze=True).transpose(0, 1).to(targets.device)
                    out_data, F1, F2 = self.model(rendered_images)
                else:
                    out_data = self.model(rendered_images)

                feat_list.append(self.model.pooled_view.cpu().numpy())
                target_list.append(targets.cpu().numpy())
        features = np.concatenate(feat_list)
        targets = np.concatenate(target_list)
        return features, targets

    def update_retrieval(self, path=None):
        all_correct_points = 0
        all_points = 0
        count = 0
        wrong_class = np.zeros(len(self.classes))
        samples_class = np.zeros(len(self.classes))
        all_loss = 0
        self.model.eval()
        self.models_bag["mvtn"].eval()
        self.models_bag["mvrenderer"].eval()
        self.models_bag["color"].eval()


        # COMPUTE THE FEATURES:
        print("Compute traiing features?")
        os.makedirs(os.path.dirname(self.setup["feature_file"]), exist_ok=True)
        if not os.path.exists(self.setup["feature_file"]) or not os.path.exists(self.setup["targets_file"]):
            print("Compute them")
            features, targets = self.compute_features()
            np.save(self.setup["feature_file"], features)
            np.save(self.setup["targets_file"], targets)

        import pickle as pkl

        # reduce Features:
        print("Fit the LDFA?")
        if not os.path.exists(self.setup["LFDA_file"]) and self.setup['LFDA_dimension'] > 0:
            print("Fit LDFA")
            from metric_learn import LFDA
            features = np.load(self.setup["feature_file"])
            targets = np.load(self.setup["targets_file"])
            lfda = LFDA(n_components=self.setup['LFDA_dimension'])
            lfda.fit(features, targets)
            with open(self.setup["LFDA_file"], "wb") as fobj:
                pkl.dump(lfda, fobj)

        import scipy.spatial
        features_training = np.load(self.setup["feature_file"])
        targets_training = np.load(self.setup["targets_file"])
        N_retrieved = 1000 if "shapenetcore" in self.setup["data_dir"].lower() else len(features_training)

        if self.setup['LFDA_dimension'] > 0:
            with open(self.setup["LFDA_file"], "rb") as fobj:
                lfda = pkl.load(fobj)

            features_training = lfda.transform(features_training)
        kdtree = scipy.spatial.KDTree(features_training)
        all_APs = []
        all_PNs = []
        all_RNs = []
        all_FNs = []

        print("start evaluation")
        views_record = ListDict(["azim", "elev", "dist", "label", "view_nb", "exp_id"])
        from tqdm import tqdm

        for ii, (targets, meshes, points) in tqdm(enumerate(self.val_loader)):
            with torch.no_grad():

                c_batch_size = targets.shape[0]
                targets = targets.to(self.device).long()
                points = points.to(self.device)
                targets = Variable(targets)
                azim, elev, dist = self.models_bag["mvtn"](
                    points, c_batch_size=c_batch_size)
                color = None
                if self.setup["object_color"] == "custom":
                    color = self.models_bag["color"](points.permute(0, 2, 1))

                rendered_images, _ = self.models_bag["mvrenderer"](
                    meshes, points, azim=azim, elev=elev, dist=dist, color=color)
                N, V, C, H, W = rendered_images.size()
                rendered_images = self.Normalize(rendered_images.contiguous(
                ).view(-1, C, H, W), ignore_normalize=self.setup["ignore_normalize"])

                if self.model_name == 'svcnn':
                    targets = targets.repeat_interleave(V)

                if self.model_name == 'view-gcn':
                    self.model.vertices = unbatch_tensor(
                        camera_position_from_spherical_angles(distance=batch_tensor(
                            dist.T, dim=1, squeeze=True), elevation=batch_tensor(elev.T, dim=1, squeeze=True),
                            azimuth=batch_tensor(azim.T, dim=1, squeeze=True)), batch_size=self.setup["nb_views"],
                        dim=1, unsqueeze=True).transpose(0, 1).to(targets.device)
                    out_data, F1, F2 = self.model(rendered_images)
                else:
                    out_data = self.model(rendered_images)
                pred = torch.max(out_data, 1)[1]
                all_loss += self.loss_fn(out_data, targets).cpu().data.numpy()
                results = pred == targets

                for i in range(results.size()[0]):
                    if not bool(results[i].cpu().data.numpy()):
                        wrong_class[targets.cpu().data.numpy().astype('int')[i]] += 1
                    samples_class[targets.cpu().data.numpy().astype('int')[i]] += 1
                correct_points = torch.sum(results.long())

                all_correct_points += correct_points
                all_points += results.size()[0]
                if self.model_name == 'svcnn':
                    c_views = ListDict({"azim": azim.cpu().numpy().reshape(-1).tolist(),
                                        "elev": elev.cpu().numpy().reshape(-1).tolist(),
                                        "dist": dist.cpu().numpy().reshape(-1).tolist(),
                                        "label": targets.cpu().numpy().tolist(),
                                        "view_nb": len(meshes) * list(range(self.setup["nb_views"])),
                                        "exp_id": len(meshes) * int(self.setup["nb_views"]) * [self.setup["exp_id"]]})
                else:
                    c_views = ListDict({"azim": azim.cpu().numpy().reshape(-1).tolist(),
                                        "elev": elev.cpu().numpy().reshape(-1).tolist(),
                                        "dist": dist.cpu().numpy().reshape(-1).tolist(),
                                        "label": np.repeat(targets.cpu().numpy(), self.setup["nb_views"]).tolist(),
                                        "view_nb": int(targets.cpu().numpy().shape[0]) * list(
                                            range(self.setup["nb_views"])),
                                        "exp_id": int(targets.cpu().numpy().shape[0]) * int(self.setup["nb_views"]) * [
                                            self.setup["exp_id"]]})
                views_record.extend(c_views)
                # if retrieval:
                # compute retrieval
                # batch_features = self.model.pooled_view
                feat = self.model.pooled_view.cpu().numpy()
                if self.setup['LFDA_dimension'] > 0:
                    feat = lfda.transform(feat)
                d, idx_closest = kdtree.query(feat, k=len(features_training))
                # loop over queries in the query
                for i_query_batch in range(feat.shape[0]):
                    # details on retrieval-mAP: https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52#f9ce
                    positives = targets_training[idx_closest[i_query_batch, :]] == targets[i_query_batch].cpu().numpy()
                    # AP: numerator is cumulative of positives, zero-ing negatives
                    num = np.cumsum(positives)
                    num[~positives] = 0
                    # AP: denominator is number of retrieved shapes
                    den = np.array([i + 1 for i in range(len(features_training))])
                    # AP: GTP is number of positive ground truth
                    GTP = np.sum(positives)
                    AP = np.sum(num / den) / GTP
                    PN = (np.max(num[:N_retrieved]) / N_retrieved)
                    RN = (np.max(num[:N_retrieved]) / GTP)
                    FN = 2.0 * PN * RN / (PN + RN + 0.000001)
                    # if np.sum(num[:10]) > 3:
                    #     want.append(ii)
                    # print("closeset of {} from class {}".format(ii, targets.cpu().numpy()[0]))
                    # for kk in range(10):
                    #     print(idx_closest[0][kk], num[kk], ~positives[kk])
                    # print(N_retrieved, AP,PN,RN,FN)
                    all_APs.append(AP)
                    all_PNs.append(PN)
                    all_RNs.append(RN)
                    all_FNs.append(FN)

        # print(want,"\n",all_APs)
        print('Total # of test models: ', all_points)
        class_acc = (samples_class - wrong_class) / samples_class
        val_mean_class_acc = np.mean(class_acc)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)

        print('val mean class acc. : ', val_mean_class_acc)
        print('val overall acc. : ', val_overall_acc)
        print('val loss : ', loss)
        print(class_acc)
        self.model.train()
        self.models_bag["mvtn"].train()
        self.models_bag["color"].train()
        self.models_bag["mvrenderer"].train()
        if self.model_name == 'view-gcn' and self.setup["no_joint_train"]:
            self.models_bag["color"].eval()

        retr_map = 100 * sum(all_APs) / len(all_APs)
        retr_pn = 100 * sum(all_PNs) / len(all_PNs)
        retr_rn = 100 * sum(all_RNs) / len(all_RNs)
        retr_fn = 100 * sum(all_FNs) / len(all_FNs)

        # print("avg_loss", avg_loss)
        # print("avg_test_acc", avg_test_acc)
        print("retr_map", retr_map)
        print("retr_pn", retr_pn)
        print("retr_rn", retr_rn)
        print("retr_fn", retr_fn)
        # setup_dict = ListDict(list(self.setup.keys()))
        # save_results(self.setup["results_file"], setup_dict.append(self.setup))

        if self.setup["log_metrics"]:
            self.writer.add_hparams(self.setup, {
                "hparams/test_acc": val_overall_acc.item(),
                "hparams/best_cls_avg_acc": val_mean_class_acc.item(),
                "hparams/retr_map": retr_map,
                "hparams/retr_pn": retr_pn,
                "hparams/retr_rn": retr_rn,
                "hparams/retr_fn": retr_fn,

            })
        return retr_map, retr_pn, retr_rn, retr_fn




