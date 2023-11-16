import evaluation
import factories
from dataset.captioning_dataset import CaptioningDataset
from evaluation import PTBTokenizer


def test_given_example_input_when_eval_then_compute_global_scores():
    # Arrange
    tokeniser = PTBTokenizer()
    val_data = CaptioningDataset(
        "/import/gameai-01/eey362/datasets/coco/images",
        "/homes/hps01/ViGCap/data/karpathy_splits/dataset_coco.json",
        dataset_name="coco",
        freq_threshold=5,
        split="val",
    )
    val_dataloader = factories.get_dataloader(val_data, 32, shuffle=False)
    _, _, caps_gt = next(iter(val_dataloader))

    # Act
    gen = {}
    gts = {}
    for it, (_, _, caps_gt) in enumerate(val_dataloader):
        caps_gen = [cap[0] for cap in caps_gt]

        for i in range(len(caps_gt)):
            gen[f"{it}_{i}"] = [caps_gen[i]]
            gts[f"{it}_{i}"] = [caption for caption in caps_gt[i]]

    predictions = tokeniser.tokenize(gen)
    references = tokeniser.tokenize(gts)

    scores, _ = evaluation.compute_scores(references, predictions)

    # Assert
    assert scores["CIDEr"] > 1.00
