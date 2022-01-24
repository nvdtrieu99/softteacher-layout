_base_="base.py"

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(

        sup=dict(

            ann_file="../../input/v24-soft-teacher/annotations/train_v2.4.json",
            img_prefix="../../input/v24-soft-teacher/train/",

        ),
        unsup=dict(

            ann_file="../../input/v24-soft-teacher/annotations/train_v2.4-unlabeled.json",
            img_prefix="../../input/v24-soft-teacher/train/",

        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 1],
        )
    ),
)

semi_wrapper = dict(
    train_cfg=dict(
        unsup_weight=2.0,
    )
)

lr_config = dict(step=[120000 * 4, 160000 * 4])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000 * 4)

