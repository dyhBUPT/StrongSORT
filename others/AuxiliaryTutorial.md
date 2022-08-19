# Auxiliary tutorial

## 1. How to generate ECC results?

Just run the "others/ecc.py".

You should modify the path in line [150](https://github.com/dyhBUPT/StrongSORT/blob/5c2d1d0ba22bf0d0440e217c60554e5e56f93ac0/others/ecc.py#L150) and [178](https://github.com/dyhBUPT/StrongSORT/blob/5c2d1d0ba22bf0d0440e217c60554e5e56f93ac0/others/ecc.py#L178).

## 2. How to generate detections?

The easiest way to generate detections on MOT17 and MOT20 is using the source code of [ByteTrack](https://github.com/ifzhang/ByteTrack).

You should modify the code in "yolox/evaluators/mot_evaluator.py" in ByteTrack as follows (We have provided our modified code in "others/mot_evaluator.py"):

1. Set NMS parameters.

   ```python
   # outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
   outputs = postprocess(outputs, self.num_classes, .6, .8)
   ```

2. Comment out the tracking code.

   ```python
   # run tracking
   # if outputs[0] is not None:
   #     online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
   #     online_tlwhs = []
   #     online_ids = []
   #     online_scores = []
   #     for t in online_targets:
   #         tlwh = t.tlwh
   #         tid = t.track_id
   #         vertical = tlwh[2] / tlwh[3] > 1.6
   #         if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
   #             online_tlwhs.append(tlwh)
   #             online_ids.append(tid)
   #             online_scores.append(t.score)
   #     # save results
   #     results.append((frame_id, online_tlwhs, online_ids, online_scores))
   #
   # if is_time_record:
   #     track_end = time_synchronized()
   #     track_time += track_end - infer_end
   #
   # if cur_iter == len(self.dataloader) - 1:
   #     result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
   #     write_results(result_filename, results)
   ```

3. Save detection results.

   ```python
   # eval_results = self.evaluate_prediction(data_list, statistics)
   json.dump(data_list, open(r'./YOLOX_xxx.json', 'w'))
   ```

Then, you can generate detections by running the ByteTrack. For example:

```shell
python tools/track.py \
-f exps/example/mot/yolox_x_ablation.py \
-c xxx/bytetrack_ablation.pth.tar \
-b 1 \
-d 1 \
--fp16 \
--fuse
```

## 3. How to generate features?

We generate detection features based on [FastReID](https://github.com/JDAI-CV/fast-reid).

Please follow its tutorial to train a feature extractor.

Then you can use it to extract detection features. Here you need to write a simple script.

We also give an example code in "ohters/generate_detections.py" for reference only, which takes TXT format detections file as input. 

## 4. Custom Dataset.

To run StrongSORT on the custom dataset, we provide a coarse guide as follows:

- Prepare your dataset as the format of MOTChallenge, like MOT17.

- Prepare the StrongSORT referring to the README. Please remember to modify the data path in "opt.py". Then, adding the data infomation of your dataset in the "data" (dict) in "opt.py".

- Prepare the results of detections、features and ECC results(optional) referring to the item 1, 2, 3.

Then, you can try to run the StrongSORT referring to the README. Please Note that the "ECC" is not nessary if there is no movements of cameras in your dataset.
