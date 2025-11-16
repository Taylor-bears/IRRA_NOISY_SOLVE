import os.path as op
from typing import List

from utils.iotools import read_json
from .bases import BaseDataset

"""
核心功能包括：
​​解析数据集标注文件​​（JSON格式），按 train/test/val划分数据。
​​构建结构化数据​​：生成图像-文本对，并提取身份标签（PID）。
​​验证数据完整性​​：检查文件路径是否存在。
"""

class CUHKPEDES(BaseDataset):
    """
    CUHK-PEDES

    Reference:
    Person Search With Natural Language Description (CVPR 2017)

    URL: https://openaccess.thecvf.com/content_cvpr_2017/html/Li_Person_Search_With_CVPR_2017_paper.html

    Dataset statistics:
    ### identities: 13003
    ### images: 40206,  (train)  (test)  (val)
    ### captions: 
    ### 9 images have more than 2 captions
    ### 4 identity have only one image

    annotation format: 
    [{'split', str,
      'captions', list,
      'file_path', str,
      'processed_tokens', list,
      'id', int}...]
    """
    dataset_dir = 'CUHK-PEDES'

    def __init__(self, root="", reid_raw= 'reid_raw.json', test_noisy_json: str | None = None, verbose=True):
        super(CUHKPEDES, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.img_dir = op.join(self.dataset_dir, 'imgs/')

        self.anno_path = op.join(self.dataset_dir, reid_raw)
        self._check_before_run()

        self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)
        # 若提供了带噪声标注的测试集JSON，则读取对照测试集（顺序与数量与 reid_raw 保持一致）
        if test_noisy_json:
            try:
                #  self.test_annos 沿用reid_raw里的分割得到，因为那个里面的test是含有20%/50%/80%比例噪声的文本，而不是全部都是噪声的文本
                noisy_annos = read_json(test_noisy_json) # compare_annos 包含每对干净与噪声文本
                self.test_compare_annos = [a for a in noisy_annos if a.get('split', 'test') == 'test']
                self.logger.info(f"Loaded compare test json: {test_noisy_json} with {len(self.test_compare_annos)} entries")
            except Exception as e:
                self.logger.warning(f"Failed to read test noisy json '{test_noisy_json}': {e}. Fallback to default test from reid_raw.json")

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        if test_noisy_json:
            self.test, self.test_id_container = self._process_anno(self.test_annos, test_compare_annos=self.test_compare_annos)
        else:
            self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            self.logger.info("=> CUHK-PEDES Images and Captions are loaded")
            self.show_dataset_info()


    def _split_anno(self, anno_path: str):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)
        for anno in annos:
            if anno['split'] == 'train':
                train_annos.append(anno)
            elif anno['split'] == 'test':
                test_annos.append(anno)
            else:
                val_annos.append(anno)
        return train_annos, test_annos, val_annos


    def _process_anno(self, annos: List[dict], training=False, test_compare_annos: List[dict] | None = None):
        pid_container = set()
        assert len(test_compare_annos) == len(self.test_annos)
        if training:
            dataset = []
            image_id = 0
            for anno in annos:
                pid = int(anno['id']) - 1 # make pid begin from 0
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['file_path'])
                captions = anno['captions'] # caption list
                for caption in captions:
                    dataset.append((pid, image_id, img_path, caption))
                image_id += 1
            for idx, pid in enumerate(sorted(pid_container)):
                # check pid begin from 0 and no break（排序后校验，避免set遍历无序）
                assert idx == pid, f"idx: {idx} and pid: {pid} are not match"
            return dataset, pid_container
        else:
            dataset = {}
            img_paths = []
            image_pids = []
            # 两种文本集合：noisy（默认用于 captions）、clean（clean_captions）
            noisy_captions = []
            clean_captions = []
            caption_pids = []
            for index, anno in enumerate(annos):
                assert test_compare_annos[index]['file_path'] == self.test_annos[index]['file_path']
                assert len(self.test_annos[index]['captions']) == len(test_compare_annos[index]['captions'])
                pid = int(anno['id'])
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['file_path'])
                img_paths.append(img_path)
                image_pids.append(pid)
                # 兼容两种格式：仅 captions（原始），或 captions + captions_rw（带噪）
                cap_list_noisy = anno.get('captions', None)
                if test_compare_annos is not None:
                    cap_list_clean = test_compare_annos[index].get('captions', []) or []
                else:
                    cap_list_clean = []
                if cap_list_noisy is None:
                    cap_list_noisy = cap_list_clean
                for c in cap_list_noisy:
                    noisy_captions.append(c)
                    caption_pids.append(pid)
                for c in cap_list_clean:
                    clean_captions.append(c)
            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                # noisy 版本作为默认 captions
                "captions": noisy_captions,
                # clean 版本
                "clean_captions": clean_captions if clean_captions else noisy_captions,
            }
            return dataset, pid_container


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not op.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))
