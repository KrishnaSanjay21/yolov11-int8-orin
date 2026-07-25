from qint.coco import load_coco_gt


def _toy():
    return {
        "categories": [{"id": 5, "name": "cat"}, {"id": 2, "name": "dog"}],
        "images": [
            {"id": 100, "file_name": "a.jpg", "height": 480, "width": 640},
            {"id": 101, "file_name": "b.jpg", "height": 720, "width": 1280},
        ],
        "annotations": [
            {"image_id": 100, "category_id": 5, "bbox": [10, 20, 30, 40], "iscrowd": 0},
            {"image_id": 101, "category_id": 2, "bbox": [0, 0, 5, 5]},
            {"image_id": 100, "category_id": 2, "bbox": [1, 1, 2, 2], "iscrowd": 1},  # skipped
        ],
    }


def test_dense_remap_and_names():
    gts, names, images = load_coco_gt(_toy())
    # categories sorted by id -> [dog(2), cat(5)]
    assert names == ["dog", "cat"]
    assert images[0] == ("a.jpg", (480, 640))
    assert images[1] == ("b.jpg", (720, 1280))


def test_bbox_xywh_to_xyxy_and_crowd_filtered():
    gts, names, _ = load_coco_gt(_toy())
    assert len(gts) == 2  # iscrowd annotation dropped
    g0 = [g for g in gts if g.image_id == 0][0]
    assert g0.class_id == 1  # cat -> dense id 1
    assert g0.box == (10.0, 20.0, 40.0, 60.0)  # x,y,x+w,y+h
