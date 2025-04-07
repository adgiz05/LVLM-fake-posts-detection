INT2LABEL = {
    '2_way_label': {0: 'fake', 1: 'true'},
    '3_way_label': {0: 'true', 1: 'fake', 2: 'partially true'},
    '6_way_label': {0: 'true', 1: 'satire/parody', 2: 'misleading content', 3: 'imposter content', 4: 'false_connection', 5: 'manipulated content'}
}

CLS2SUBREDDIT = {
    0: 'mildlyinteresting',
    1: 'pareidolia',
    2: 'neutralnews',
    3: 'photoshopbattles',
    4: 'nottheonion',
    5: 'psbattle_artwork',
    6: 'fakehistoryporn',
    7: 'propagandaposters',
    8: 'upliftingnews',
    9: 'fakealbumcovers',
    10: 'subredditsimulator',
    11: 'satire',
    12: 'savedyouaclick',
    13: 'misleadingthumbnails',
    14: 'pic',
    15: 'theonion',
    16: 'confusing_perspective',
    17: 'usnews',
    18: 'usanews',
    19: 'subsimulatorgpt2',
    20: 'waterfordwhispersnews',
    21: 'fakefacts'
}

SUBREDDIT2CLS = {v: k for k, v in CLS2SUBREDDIT.items()}

TRAIN_GRAPH_DICT_PATH   = "dataset/comments/train_comment_trees.pkl"
VAL_GRAPH_DICT_PATH     = "dataset/comments/val_comment_trees.pkl"
TEST_GRAPH_DICT_PATH    = "dataset/comments/test_comment_trees.pkl"
IMAGES_PATH             = "dataset/images"