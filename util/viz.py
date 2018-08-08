from collections import Counter, defaultdict
import os.path as osp

from graphviz import Digraph
import matplotlib.pyplot as plt
import numpy as np

from px2graph_lab.util.colormap import colormap


def visualize_preds(inp_img, obj_preds, rel_preds, dataset, output_name, output_dir,
                    obj_score_th=0.3, dpi=200, box_alpha=0.3, ext='pdf'):
    obj_preds = obj_preds[obj_preds[:, 0] > 0]
    obj_scores, obj_classes, _, obj_bboxes, _, obj_labels = np.split(obj_preds, [1, 2, 11, 15, 23], axis=1)
    obj_classes = np.squeeze(obj_classes.astype(int), axis=1)
    obj_labels = np.squeeze(obj_labels.astype(int), axis=1)  # FP(0), TP(1), FN(2)

    rel_preds = rel_preds[rel_preds[:, 0] > 0]
    rel_scores, rel_sbjs, rel_classes, rel_objs, rel_labels = np.split(rel_preds, 5, axis=1)
    rel_classes = np.squeeze(rel_classes.astype(int), axis=1)
    rel_sbjs = np.squeeze(rel_sbjs.astype(int), axis=1)
    rel_objs = np.squeeze(rel_objs.astype(int), axis=1)
    rel_labels = np.squeeze(rel_labels.astype(int), axis=1)

    num_rels = len(dataset.relationships)

    # Display in largest to smallest order to reduce occlusion
    areas = (obj_bboxes[:, 2] - obj_bboxes[:, 0]) * (obj_bboxes[:, 3] - obj_bboxes[:, 1])
    sorted_inds = np.argsort(-areas)

    nodes = []
    edges = []
    objId2arrIdx = {}
    arrIdx = 0
    for ind in sorted_inds:
        if obj_scores[ind] > obj_score_th:
            nodes.append(ind)
            objId2arrIdx[ind] = arrIdx
            arrIdx += 1
    node_set = set(nodes)
    node_ref_by_rel = [False]*len(nodes)
    has_dup_rel = defaultdict(bool)
    for ind, (rel_cls, sbj, obj) in enumerate(zip(rel_classes, rel_sbjs, rel_objs)):
        if rel_cls < num_rels and sbj in node_set and obj in node_set and \
                not has_dup_rel[(rel_cls, sbj, obj)]:
            has_dup_rel[(rel_cls, sbj, obj)] = True
            edges.append(ind)
            node_ref_by_rel[objId2arrIdx[sbj]] = True
            node_ref_by_rel[objId2arrIdx[obj]] = True
    nodes = np.array(nodes)[node_ref_by_rel]

    color_list = colormap(rgb=True) / 255
    color_ind = 0
    fig = plt.figure(frameon=False)
    fig.set_size_inches(inp_img.shape[1] / dpi, inp_img.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(inp_img)

    u = Digraph(osp.join(output_dir, output_name + '_sg'))
    u.body.append('size="6,6"')
    u.body.append('rankdir="LR"')

    objcls_counter = Counter()

    for obj_ind in nodes:
        color = color_list[color_ind % len(color_list), :3]
        color_ind += 1
        bbox = obj_bboxes[obj_ind]
        cls_id = obj_classes[obj_ind]
        cls_name = dataset.class_labels[cls_id]
        cnt = objcls_counter[cls_name]
        obj_name = '%s_%d' % (cls_name, cnt) if cnt else cls_name
        objcls_counter.update([cls_name])
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                          fill=False, edgecolor=color, linewidth=1, alpha=box_alpha)
        )
        ax.text(
            bbox[0] - 1, bbox[1] - 3, obj_name, fontsize=3, family='serif',
            bbox=dict(
                facecolor=color, alpha=0.4, pad=0, fc=color, lw=0
            )
        )
        colorhex = '#'
        for n in color:
            h = hex(int(n*255))[2:]
            if len(h) == 1:
                h = '0' + h
            colorhex += h
        u.node(str(obj_ind), label=obj_name, color=colorhex, penwidth="2")
    out_imgname = output_name + '_bbox.' + ext
    fig.savefig(osp.join(output_dir, out_imgname), dpi=dpi)
    plt.close('all')

    for ind in edges:
        sbj_id, obj_id, cls_id = rel_sbjs[ind], rel_objs[ind], rel_classes[ind]
        edge_key = '%d_%d_%d' % (sbj_id, obj_id, cls_id)
        u.node(edge_key, label=dataset.relationships[cls_id],
               color='gray', shape='box', style='filled')
        u.edge(str(sbj_id), edge_key)
        u.edge(edge_key, str(obj_id))
    u.render()
