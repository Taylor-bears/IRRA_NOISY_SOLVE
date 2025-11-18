import re
from typing import List, Tuple
import nltk
import os

# 优先从项目本地 nltk_data 目录加载，避免训练时联网下载
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_NLTK_LOCAL_DIR = os.getenv('NLTK_DATA', os.path.join(_PROJECT_ROOT, 'nltk_data'))
if _NLTK_LOCAL_DIR not in nltk.data.path:
    nltk.data.path.insert(0, _NLTK_LOCAL_DIR)

# 需要的资源：分词 + 词形还原(wordnet) + 词性标注
def _ensure_pkg(pkg: str):
    try:
        if pkg == 'punkt':
            nltk.data.find('tokenizers/punkt')
        elif 'tagger' in pkg:
            nltk.data.find(f'taggers/{pkg}')
        else:
            nltk.data.find(f'corpora/{pkg}')
    except LookupError:
        try:
            os.makedirs(_NLTK_LOCAL_DIR, exist_ok=True)
            nltk.download(pkg, download_dir=_NLTK_LOCAL_DIR)
        except Exception:
            # 若离线则静默，后续会走回退逻辑
            pass

for _pkg in ['punkt','wordnet','omw-1.4']:
    _ensure_pkg(_pkg)
# 优先使用新的英文感知版感知器标注器；若不存在则回退旧版
_POS_RESOURCE_OK = True # 词性标注资源
try:
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    # 定向下载至本地目录
    try:
        _ensure_pkg('averaged_perceptron_tagger_eng')
    except Exception:
        try:
            _ensure_pkg('averaged_perceptron_tagger')
        except Exception:
            _POS_RESOURCE_OK = False
from nltk.stem import WordNetLemmatizer
_LEMMATIZER = WordNetLemmatizer() # 词形还原器
_NLTK_OK = True


# 基础可视属性词典（融合 prompt 中涉及的类别）
# 颜色词集合
_COLOR_WORDS = {
    'black', 'white', 'red', 'blue', 'green', 'yellow', 'gray', 'grey', 'pink', 'purple', 'brown', 'orange', 'navy', 'beige', 'gold', 'golden', 'silver','light', 'dark', 'bright', 'pale', 'deep', 'faded','scarlet', 'emerald', 'lavender', 'fuchsia', 'azure', 'turquoise', 'beetroot', 'cobalt', 'charcoal', 'coral', 'mint', 'burgundy', 'cream', 'chocolate', 'indigo', 'magenta', 'plum', 'copper', 'ivory'
}
# 服装词集合
_CLOTH_WORDS = {
    'jacket', 'coat', 'hoodie', 'shirt', 'tshirt', 't-shirt', 'trousers', 'pants', 'jeans', 'skirt', 'shorts', 'dress', 'sneakers', 'shoes', 'boots', 'sandals', 'cardigan', 'blouse', 'sweater', 'vest', 'blazer', 'suit', 'pajamas', 'scarf', 'tie', 'gloves', 'hat', 'cap', 'beret', 'belt', 'leggings', 'overalls', 'romper', 'bikini', 'swimsuit', 'maternity dress', 'robe', 'gown', 'culottes', 'bodysuit', 'kimono', 'trench coat', 'parka', 'denim jacket', 'flannel shirt'
}
# 配饰词集合
_ACCESSORY_WORDS = {
    'backpack', 'handbag', 'bag', 'tote', 'watch', 'bracelet', 'necklace', 'ring', 'glasses', 'sunglasses', 'hat', 'cap', 'scarf', 'belt', 'wallet', 'purse', 'earrings', 'brooch', 'cufflinks', 'headband', 'anklet','bandana', 'bangle', 'choker', 'keychain', 'fanny pack', 'gloves', 'tie clip', 'hairpin', 'hairclip', 'suspenders', 'bow tie', 'lip balm', 'umbrella'
}
# 发型发色词集合
_HAIR_WORDS = {
    'hair', 'ponytail', 'bangs', 'blond', 'blonde', 'brown', 'black', 'short', 'long', 'curly', 'straight', 'wavy', 'frizzy', 'bald', 'bob', 'pixie', 'braid', 'bun', 'shaved', 'faded', 'afro', 'dreadlocks', 'buzzcut', 'updo', 'layers', 'highlighted', 'streaked', 'tousled', 'plaited', 'side-parted', 'center-parted', 'choppy', 'slicked-back', 'mullet'
}
# 图案纹理词集合
_PATTERN_WORDS = {
    'striped', 'plain', 'denim', 'leather', 'plaid', 'floral', 'checkered', 'patterned', 'polka dot', 'herringbone', 'paisley', 'camouflage', 'tartan', 'ribbed', 'quilted', 'houndstooth', 'jacquard', 'suede', 'corduroy', 'brocade', 'geometric', 'abstract', 'animal print','zebra print', 'leopard print', 'snake print', 'plaid', 'tie-dye', 'textured', 'woven'
}
# 物品词集合
_ITEM_WORDS = {
    'phone', 'book', 'camera', 'umbrella', 'cup', 'laptop', 'tablet', 'headphones', 'wallet', 'key', 'notebook', 'pen', 'pencil', 'bag', 'backpack', 'watch', 'sunglasses', 'gloves', 'umbrella', 'hat', 'scarf', 'shoes', 'glasses', 'charger', 'flashlight', 'speaker', 'toothbrush', 'toothpaste', 'lipstick', 'comb', 'mirror', 'scissors', 'towel', 'bottle','blanket', 'wallet', 'broom', 'vacuum', 'knife', 'fork', 'spoon', 'plate', 'cup', 'bowl', 'dish', 'mug', 'pan', 'pot', 'clothes', 'shoes', 'slippers', 'scissors', 'tissue'
}

# 合并所有词典创建一个完整的属性词库
_LEXICON = set().union(_COLOR_WORDS,_CLOTH_WORDS,_ACCESSORY_WORDS,_HAIR_WORDS,_PATTERN_WORDS,_ITEM_WORDS)

# 多词短语（所有元素均需顺序匹配；用于提升组合属性覆盖）
# 注意：这些短语的组成词中有的单词本身不是属性（如 shoulder），需要借助短语整体判定。
_MULTI_WORD_PHRASES = [
    ('shoulder', 'bag'),    # 单肩包
    ('coffee', 'cup'),      # 咖啡杯
    ('smart', 'phone'),     # 智能手机
    ('sun', 'glasses'),     # 太阳镜
    ('wrist', 'watch'),     # 手腕表
    ('wool', 'scarf'),      # 羊毛围巾
    ('leather', 'jacket'),  # 皮夹克
    ('red', 'dress'),       # 红色连衣裙
    ('sports', 'shoes'),    # 运动鞋
    ('leopard', 'print'),   # 豹纹
    ('leather', 'boots'),   # 皮靴
    ('high', 'heels'),      # 高跟鞋
    ('designer', 'bag'),    # 名牌包
    ('winter', 'coat'),     # 冬季大衣
    ('cotton', 'shirt'),    # 棉质衬衫
    ('gold', 'bracelet'),   # 黄金手链
    ('silver', 'necklace'), # 银项链
    ('denim', 'jeans'),     # 牛仔裤
    ('plaid', 'skirt'),     # 格子裙
    ('wool', 'sweater'),    # 羊毛毛衣
]

# 归一化后的短语集合，便于常数时间匹配
_PHRASE_SET = set()
for a,b in _MULTI_WORD_PHRASES:
    na = re.sub(r"[^A-Za-z]+","", a).lower()
    nb = re.sub(r"[^A-Za-z]+","", b).lower()
    if na and nb:
        _PHRASE_SET.add((na, nb))

# 简单停用词/基础词集合（不参与组合短语/邻接判定）
_STOPWORDS = {
    'a','an','the','and','or','but','with','without','of','on','in','to','from','by','for','as','at',
    'is','are','am','was','were','be','been','being','this','that','these','those','there','here',
    'very','really','just','so','over','under','between','into','onto','off','up','down','out','inside','outside','near','far',
    'he','she','it','they','you','we','his','her','their','them','him','hers','ours','your','my','mine','our','yours','me','i'
}

# 不再使用停用词/词性过滤。
def _normalize_word(w: str) -> str: # 将单词归一化为基本形式
    base = re.sub(r"[^A-Za-z]+","", w).lower()
    if not base:
        return ''
    if not _NLTK_OK:
        # 简单复数去尾处理（非常粗糙，NLTK不可用时退化）
        if base.endswith('es') and len(base) > 3:
            return base[:-2]
        if base.endswith('s') and len(base) > 3:
            return base[:-1]
        return base
    # 依次按名词/形容词/动词进行词形还原，选择最短的结果作为标准形式
    forms = [
        _LEMMATIZER.lemmatize(base, pos='n'),
        _LEMMATIZER.lemmatize(base, pos='a'),
        _LEMMATIZER.lemmatize(base, pos='v'),
    ]
    # 选取最短长度（通常为主词根）
    norm = min(forms, key=len)
    return norm

# 是否为基础词/停用词 
def _is_basic_word(base: str) -> bool:
    return base in _STOPWORDS if base else False

# 输入词序列，返回对应词性标注列表
def _get_pos_tags(words: List[str]) -> List[str]:
    """返回与 words 等长的词性标注列表；若资源不可用则进行简易回退。"""
    tags: List[str] = []
    if _POS_RESOURCE_OK:
        try:
            tags = [t for _, t in nltk.pos_tag(words)]
        except Exception:
            # 发生运行时错误则回退为简易规则
            tags = []
    if not tags:
        # 简易回退：已知颜色/常见形容词视作 JJ；词典词默认 NN；其余 NN
        for w in words:
            b = _normalize_word(w)
            if b in _COLOR_WORDS or b in _HAIR_WORDS or b in _PATTERN_WORDS or b.endswith('y'):
                tags.append('JJ')
            elif b in _LEXICON:
                tags.append('NN')
            else:
                tags.append('NN')
    return tags

# 通过单个词的词典判断
def is_attribute_word(word: str) -> bool:
    """单词级判定：仅检查是否在词典中。保留用于需要无上下文快速判断的场景。"""
    base = _normalize_word(word)
    return base in _LEXICON if base else False

# 通过上下文判定某词是否组成了短语来判断
def _is_attribute_token_with_context(words: List[str], tags: List[str], i: int) -> bool:
    """根据你的规则进行上下文判定：
    - 若当前词在词典中：True。
    - 若当前为形容词(JJ)，其前或后存在“在词典中的名词(NN)”：True。
    - 若当前为名词(NN)但不在词典中，与相邻词组成的二词短语在短语表：True。
    - 基础词/停用词不参与组合检查。
    """
    n = len(words)
    base = _normalize_word(words[i])
    if not base or _is_basic_word(base):
        return False
    # 直接命中词典
    if base in _LEXICON:
        return True
    tag = tags[i] if 0 <= i < len(tags) else 'NN'
    is_adj = tag.startswith('JJ')
    is_noun = tag.startswith('NN')

    # 形容词：前/后邻为词典名词
    if is_adj:
        for j in (i-1, i+1):
            if 0 <= j < n:
                nb = _normalize_word(words[j])
                if nb and not _is_basic_word(nb):
                    ntag = tags[j] if 0 <= j < len(tags) else 'NN'
                    if ntag.startswith('NN') and nb in _LEXICON:
                        return True
        return False

    # 名词但不在词典：尝试与相邻词组成短语
    if is_noun:
        prev_b = _normalize_word(words[i-1]) if i-1 >= 0 else ''
        next_b = _normalize_word(words[i+1]) if i+1 < n else ''
        if prev_b and not _is_basic_word(prev_b) and (prev_b, base) in _PHRASE_SET:
            return True
        if next_b and not _is_basic_word(next_b) and (base, next_b) in _PHRASE_SET:
            return True
        return False

    return False

# 返回句子中被判定为“属性词/短语”的词索引集合
def attribute_indices(words: List[str]) -> List[int]:
    """"
    应用规则：词典命中；形容词邻接词典名词；名词与相邻词构成短语命中。
    若命中短语，返回中将包含两个词的索引。
    """
    tags = _get_pos_tags(words)
    indices_set = set()
    n = len(words)
    for i in range(n):
        base = _normalize_word(words[i])
        if not base or _is_basic_word(base):
            continue
        # 直接词典命中
        if base in _LEXICON:
            indices_set.add(i)
            continue
        # 上下文规则判定
        if _is_attribute_token_with_context(words, tags, i):
            indices_set.add(i)
            # 若由短语触发，尽量把伙伴词也加入，便于掩码覆盖完整短语
            # 前向短语
            if i+1 < n:
                nb = _normalize_word(words[i+1])
                if nb and not _is_basic_word(nb) and (base, nb) in _PHRASE_SET:
                    indices_set.add(i+1)
            # 后向短语
            if i-1 >= 0:
                pb = _normalize_word(words[i-1])
                if pb and not _is_basic_word(pb) and (pb, base) in _PHRASE_SET:
                    indices_set.add(i-1)
    return sorted(indices_set)

# 根据词级属性词索引与对应token span构建token级掩码
def build_attribute_mask(words: List[str], spans: List[Tuple[int,int]], text_length: int, offset:int=1) -> 'list[int]':
    """根据词级属性词索引与对应 token span 构建 token 级掩码。
    spans: 与 words 对齐的 (start,end) token 下标（不含 offset 前的特殊 token）。
    offset: token 序列中词起始偏移（例如首位是 <sot>）。
    返回长度 text_length 的 0/1 列表。
    """
    att_word_indices = set(attribute_indices(words))
    mask = [0]*text_length
    for wi in att_word_indices:
        if wi < len(spans):
            s,e = spans[wi]
            ts = min(offset + s, text_length)
            te = min(offset + e, text_length)
            for t in range(ts,te):
                if t < text_length:
                    mask[t] = 1
    return mask