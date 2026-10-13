from typing import Dict, Set

from ..lon_ import BN


class U2B:
    """Unicode used as Glyph to Bengali"""

    premap = {
        "\u00e5\u00a1\u00b8": "\u00b8\u00a1\u00e5",  # \u00e5\u00a1\00b8 -> å¡¸ : \u00b8\u00a1\u00e5 -> ¸¡å #! Eg: security, virtual,
    }

    en_punctuations = {
        "\u0021",  # \u0021 -> ! (Exclamation mark)
        "\u0025",  # \u0025 -> % (Percent sign)
        "\u0028",  # \u0028 -> ( (Left Parenthesis)
        "\u0029",  # \u0029 -> ) (Right Parenthesis)
        "\u002c",  # \u002c -> , (Comma)
        "\u002d",  # \u002d -> - (Hyphen-minus)
        "\u002e",  # \u002e -> . (Full stop)
        "\u003a",  # \u003a -> : (Colon)
        "\u003f",  # \u003f -> ? (Question mark)
        "\u2018",  # \u2018 ->  (Left Single Quotation mark)
        "\u2019",  # \u2019 ->  (Right Single Quotation mark)
    }

    charmap = {
        "\u0022": BN.a,  # \u0022 -> " : অ #! MISSING SP
        "\u0022\u00e0": BN.aa,  # \u0022\u00e0 -> "à : আ
        "\u0023": BN.ii,  # \u0023 -> # : ঈ
        "\u0024": BN.uu,  # \u0024 -> $ : ঊ
        "\u0026": BN.e,  # \u0026 -> & : এ
        "\u0027": BN.ai,  # \u0027 -> ' : ঐ
        "\u002a": BN.o,  # \u002a -> * : ও
        "\u002b": BN.ao,  # \u002b -> + : ঔ
        "\u002f": f"{BN.ba}{BN.virama}",  # \u002f -> / : ব্ #! Up as []
        "\u0030": BN.zero,  # \u0030 -> 0 : ০
        "\u0031": BN.one,  # \u0031 -> 1 : ১
        "\u0032": BN.two,  # \u0032 -> 2 : ২
        "\u0033": BN.three,  # \u0033 -> 3 : ৩
        "\u0034": BN.four,  # \u0034 -> 4 : ৪
        "\u0035": BN.five,  # \u0035 -> 5 : ৫
        "\u0036": BN.six,  # \u0036 -> 6 : ৬
        "\u0037": BN.seven,  # \u0037 -> 7 : ৭
        "\u0038": BN.eight,  # \u0038 -> 8 : ৮
        "\u0039": BN.nine,  # \u0039 -> 9 : ৯
        "\u003b": BN.khanda_ta,  # \u003b -> ; : ৎ
        "\u003d": BN.tha,  # \u003d -> = :  থ
        "\u003e": BN.na,  # \u003e -> > : ন
        "\u0040": BN.visarga,  # \u0040 -> @ : ঃ
        "\u0041": BN.ka,  # \u0041 -> A : ক
        "\u0042": f"{BN.ka}{BN.virama}{BN.ka}",  # \u0042 -> B : ক্ক
        "\u0043": f"{BN.ka}{BN.virama}{BN.tta}",  # \u0043 -> C : ক্ট
        "\u0044": f"{BN.ka}{BN.virama}{BN.ta}{BN.virama}{BN.ba}",  # \u0044 -> D : ক্ত্ব
        "\u0045": f"{BN.ka}{BN.virama}{BN.ba}",  # \u0045 -> E : ক্ব
        "\u0046": f"{BN.ka}{BN.virama}{BN.ma}",  # \u0046 -> F : ক্ম
        "\u0047": f"{BN.ka}{BN.virama}{BN.sa}",  # \u0047 -> G : ক্স
        "\u0048": f"{BN.virama}{BN.ka}",  # \u0048 -> H :  ্ক #! Down as [sk]
        "\u0049": f"{BN.virama}{BN.ka}{BN.virama}{BN.ra}",  # \u0049 -> I : ্ক্র #! Down as [skr]
        "\u004a": BN.kha,  # \u004a -> J : খ
        "\u004b": BN.ga,  # \u004b -> K : গ
        "\u004c": f"{BN.ga}{BN.virama}{BN.ga}",  # \u004c -> L : গ্গ
        "\u004d": f"{BN.ga}{BN.virama}{BN.ba}",  # \u004d -> M : গ্ব
        "\u004e": f"{BN.ga}{BN.virama}",  # \u004e -> N : গ্ #! Up as [gn, gr, gl]
        "\u004f": f"{BN.ga}{BN.virama}",  # \u004f -> O : গ্ #! Left as []
        "\u0050": f"{BN.ga}{BN.v_u}",  # \u0050 -> P : গু
        "\u0051": BN.gha,  # \u0051 -> Q : ঘ
        "\u0052": BN.nga,  # \u0052 -> R : ঙ
        "\u0053": f"{BN.nga}{BN.virama}{BN.ka}",  # \u0053 -> S : ঙ্ক
        "\u0054": f"{BN.nga}{BN.virama}{BN.kha}",  # \u0054 -> T : ঙ্খ
        "\u0055": f"{BN.nga}{BN.virama}{BN.ga}",  # \u0055 -> U : ঙ্গ
        "\u0056": f"{BN.nga}{BN.virama}",  # \u0056 -> V : ঙ্ #! Left as [ng-gh]
        "\u0057": BN.ca,  # \u0057 -> W : চ
        "\u0058": f"{BN.ca}{BN.virama}{BN.nya}",  # \u0058 -> X : চ্ঞ
        "\u0059": f"{BN.ca}{BN.virama}{BN.cha}{BN.virama}{BN.ba}",  # \u0059 -> Y : চ্ছ্ব
        "\u005a": f"{BN.ca}{BN.virama}",  # \u005a -> Z : চ্ #! Left as []
        "\u005b": BN.v_i,  # \u005b -> [ : ি
        "\u005c": BN.ja,  # \u005c -> \ : জ
        "\u005d": f"{BN.mark_au}{BN.candrabindu}",  # \u005d -> ] : ৗঁ
        "\u005e": f"{BN.ja}{BN.virama}{BN.ja}{BN.virama}{BN.ba}",  # \u005e -> ^ : জ্জ্ব
        "\u005f": f"{BN.ja}{BN.virama}{BN.jha}",  # \u005f -> _ : জ্ঝ
        "\u0060": f"{BN.ja}{BN.virama}{BN.nya}",  # \u0060 -> ` : জ্ঞ
        "\u0061": f"{BN.ja}{BN.virama}{BN.ba}",  # \u0061 -> a : জ্ব
        "\u0062": f"{BN.ja}{BN.virama}{BN.ra}",  # \u0062 -> b : জ্র
        "\u0063": BN.jha,  # \u0063 -> c :  ঝ
        "\u0064": BN.nya,  # \u0064 -> d : ঞ
        "\u0065": f"{BN.nya}{BN.virama}{BN.ca}",  # \u0065 -> e : ঞ্চ
        "\u0066": f"{BN.nya}{BN.virama}{BN.cha}",  # \u0066 -> f : ঞ্ছ
        "\u0067": f"{BN.nya}{BN.virama}{BN.ja}",  # \u0067 -> g : ঞ্জ
        "\u0068": f"{BN.nya}{BN.virama}{BN.jha}",  # \u0068 -> h : ঞ্ঝ
        "\u0069": BN.tta,  # \u0069 -> i : ট
        "\u006a": f"{BN.tta}{BN.virama}{BN.tta}",  # \u006a -> j : ট্ট
        "\u006b": BN.ttha,  # \u006b -> k : ঠ
        "\u006c": BN.dda,  # \u006c -> l :  ড
        "\u006c\u00a1\u00fc": BN.u,  # \u006c\u00a1\u00fc -> l¡ü : উ
        "\u006d": f"{BN.dda}{BN.virama}{BN.dda}",  # \u006d -> m : ড্ড
        "\u006e": BN.ddha,  # \u006e -> n : ঢ
        "\u006f": BN.nna,  # \u006f -> o : ণ
        "\u0070": f"{BN.nna}{BN.virama}{BN.nna}",  # \u0070 -> p : ণ্ণ
        "\u0071": f"{BN.nna}{BN.virama}{BN.ttha}",  # \u0071 -> q : ণ্ঠ
        "\u0072": f"{BN.nna}{BN.virama}{BN.dda}",  # \u0072 -> r : ণ্ড
        "\u0073": f"{BN.nna}{BN.virama}",  # \u0073 -> s : ণ্ #! Left as []
        "\u0074": BN.ta,  # \u0074 -> t : ত
        "\u0075": f"{BN.ta}{BN.virama}{BN.ma}",  # \u0075 -> u : ত্ম
        "\u0076": f"{BN.ta}{BN.virama}{BN.ta}",  # \u0076 -> v : ত্ত
        "\u0076\u00a1\u00fb": f"{BN.ka}{BN.virama}{BN.ta}",  # \u0076\u00a1\u00fb -> v¡û : ক্ত
        "\u0077": f"{BN.ta}{BN.virama}{BN.ta}{BN.virama}{BN.ba}",  # \u0077 -> w :ত্ত্ব
        "\u0078": f"{BN.ta}{BN.virama}{BN.tha}",  # \u0078 -> x : ত্থ
        "\u0079": f"{BN.ta}{BN.virama}{BN.ra}",  # \u0076 -> y : ত্র
        "\u0079\u00fb": f"{BN.ka}{BN.virama}{BN.ra}",  # \u0076\u00fb -> yû : ক্র
        "\u007a": f"{BN.virama}{BN.ta}",  # \u007a -> z : ্ত #! Down as [nt, st, str]
        "\u007b": f"{BN.v_i}{BN.candrabindu}",  # \u007b -> { : িঁ
        "\u007c": f"{BN.virama}{BN.ta}{BN.virama}{BN.ra}",  # \u007c -> | : ্ত্র #! Down as [ntr, mtr, str]
        "\u0081": BN.w,  # \u0081 -> (invisible) : ৱ #! \x81
        "\u008f": f"{BN.na}{BN.virama}{BN.na}",  # \u008f ->  : ন্ন #! INVISIBLE
        "\u0090": f"{BN.na}{BN.virama}{BN.sa}",  # \u0090 ->  : ন্স #! INVISIBLE
        "\u009d": f"{BN.pa}{BN.virama}{BN.pa}",  # \u009d ->  : প্প #! INVISIBLE
        "\u00a3": f"{BN.virama}{BN.pha}",  # \u00a3 -> £ : ্ফ #! Ryt as [mf, lf]
        "\u00a4": BN.ba,  # \u00a4 -> ¤ : ব
        "\u00a5": f"{BN.ba}{BN.virama}{BN.tta}",  # \u00a5 -> ¥ : ব্ট
        "\u00a6": f"{BN.ba}{BN.virama}{BN.da}",  # \u00a6 -> ¦ : ব্দ
        "\u00a7": f"{BN.ba}{BN.virama}{BN.dha}",  # \u00a7 -> § : ব্ধ
        "\u00a8": f"{BN.ba}{BN.virama}{BN.jha}",  # \u00a8 -> ¨ : ব্ঝ
        "\u00a9": f"{BN.ba}{BN.virama}{BN.dda}",  # \u00a9 -> © : ব্ড
        "\u00aa": f"{BN.ba}{BN.virama}{BN.ja}",  # \u00aa -> ª : ব্জ
        "\u00ab": f"{BN.virama}{BN.ba}",  # \u00ab -> « : ্ব #! Down as [tv, tw, sv, khv, thv]
        "\u00ac": f"{BN.virama}{BN.ba}",  # \u00ac -> ¬ : ্ব #! Down as [mb, sv]
        "\u00ae": BN.bha,  # \u00ae -> ® : ভ
        "\u00af": f"{BN.bha}{BN.virama}{BN.la}",  # \u00af -> ¯ : ভ্ল
        "\u00b0": f"{BN.bha}{BN.virama}{BN.ra}",  # \u00b0 -> ° : ভ্র
        "\u00b1": f"{BN.virama}{BN.bha}",  # \u00b1 -> ± : ্ভ #! [mv]
        "\u00b2": f"{BN.virama}{BN.bha}{BN.virama}{BN.ra}",  # \u00b2 -> ² : ্ভ্র #! Down as []
        "\u00b3": BN.ma,  # \u00b3 -> ³ : ম
        "\u00b4": f"{BN.ma}{BN.virama}",  # \u00b4 -> ´ : ম্ #! [mv]
        "\u00b5": f"{BN.virama}{BN.ma}",  # \u00b5 -> µ : ্ম #! [nm, lm, sm]
        "\u00b6": f"{BN.virama}{BN.ma}",  # \u00b6 -> ¶ : ্ম #! [mm, sm]
        "\u00b8": f"{BN.virama}{BN.ya}",  # \u00b8 -> ¸ : ্য #! [*many]
        "\u00b9": BN.ra,  # \u00b9 -> ¹ : র
        "\u00ba": BN.la,  # \u00ba -> º : ল
        "\u00bb": f"{BN.la}{BN.virama}{BN.ka}",  # \u00bb -> » : ল্ক
        "\u00bc": f"{BN.la}{BN.virama}{BN.ga}",  # \u00bc -> ¼ :  ল্গ
        "\u00bd": f"{BN.la}{BN.virama}{BN.ga}{BN.v_u}",  # \u00bd -> ½ : ল্গু
        "\u00be": f"{BN.la}{BN.virama}{BN.ba}",  # \u00be -> ¾ : ল্ব
        "\u00bf": f"{BN.la}{BN.virama}{BN.pa}",  # \u00bf -> ¿ : ল্প
        "\u00c0": f"{BN.la}{BN.virama}{BN.la}",  # \u00c0 -> À : ল্ল
        "\u00c1": f"{BN.la}{BN.virama}{BN.dda}",  # \u00c1 -> Á : ল্ড
        "\u00c2": f"{BN.la}{BN.virama}",  # \u00c2 -> Â : ল্ #! Left as [lt, lm]
        "\u00c3": f"{BN.virama}{BN.la}",  # \u00c3 -> Ã : ্ল #! Down as [kl, gl, ml, pl, bl]
        "\u00c4": f"{BN.la}{BN.virama}",  # \u00c4 -> Ä : ল্ #! Left as [lf]
        "\u00c5": BN.sha,  # \u00c5 -> Å : শ
        "\u00c6": f"{BN.sha}{BN.virama}",  # \u00c6 -> Æ : শ্ #! Left as [sm]
        "\u00c7": f"{BN.sha}{BN.v_u}",  # \u00c7 -> Ç : শু
        "\u00c8": BN.ssa,  # \u00c8 -> È : ষ
        "\u00c9": f"{BN.ssa}{BN.virama}{BN.ba}",  # \u00c9 -> É : ষ্ব
        "\u00ca": f"{BN.ssa}{BN.virama}{BN.tta}",  # \u00ca -> Ê : ষ্ট
        "\u00cb": f"{BN.ssa}{BN.virama}{BN.ttha}",  # \u00cb -> Ë : ষ্ঠ
        "\u00cc": f"{BN.ssa}{BN.virama}{BN.nna}",  # \u00cc -> Ì : ষ্ণ
        "\u00cd": f"{BN.ssa}{BN.virama}",  # \u00cd -> Í : ষ্ #! Up as []
        "\u00ce": BN.sa,  # \u00ce -> Î : স
        "\u00cf": f"{BN.sa}{BN.virama}{BN.kha}",  # \u00cf -> Ï : স্খ
        "\u00d0": f"{BN.sa}{BN.virama}{BN.tta}",  # \u00d0 -> Ð : স্ট
        "\u00d1": f"{BN.sa}{BN.virama}",  # \u00d1 -> Ñ : স্ #! Left/Up as [sp, sk, sv, skr, st, sr]
        "\u00d2": BN.h,  # \u00d2 -> Ò : হ
        "\u00d2\u00fc": BN.i,  # \u00d2\u00fc -> Òü : ই
        "\u00d3": f"{BN.h}{BN.virama}{BN.la}",  # \u00d3 -> Ó : হ্ল
        "\u00d4": f"{BN.h}{BN.virama}{BN.ba}",  # \u00d4 -> Ô :  হ্ব
        "\u00d5": f"{BN.h}{BN.virama}{BN.ma}",  # \u00d5 -> Õ : হ্ম
        "\u00d6": f"{BN.h}{BN.virama}{BN.nna}",  # \u00d6 -> Ö : হ্ণ
        "\u00d7": f"{BN.h}{BN.v_u}",  # \u00d7 -> × : হু
        "\u00d8": BN.rra,  # \u00d8 -> Ø : ড়
        "\u00d9": BN.rha,  # \u00d9 -> Ù : ঢ়
        "\u00da": BN.yya,  # \u00da -> Ú : য়
        "\u00db": f"{BN.ka}{BN.virama}{BN.ssa}",  # \u00db -> Û : ক্ষ
        "\u00dc": f"{BN.ka}{BN.virama}{BN.ssa}{BN.virama}{BN.ma}",  # \u00dc -> Ü : ক্ষ্ম
        "\u00dd": f"{BN.ka}{BN.virama}{BN.ssa}{BN.virama}{BN.na}",  # \u00dd -> Ý : ক্ষ্ন
        "\u00de\u00ea": f"{BN.na}{BN.virama}{BN.dha}",  # \u00de\u00ea -> Þê : ন্ধ
        "\u00de\u00f8\u00fd": f"{BN.na}{BN.virama}{BN.dha}{BN.virama}{BN.ra}",  # \u00de\u00f8\u00fd -> Þøý : ন্ধ্র
        "\u00df": f"{BN.pa}{BN.virama}{BN.ra}",  # \u00df -> ß : প্র
        "\u00e0": BN.v_aa,  # \u00e0 -> à : া
        "\u00e1": BN.cha,  # \u00e1 -> á : ছ
        "\u00e2": f"{BN.ta}{BN.virama}",  # \u00e2 -> â : ত্ #! UP as [tn, tv]
        "\u00e3": BN.v_ii,  # \u00e3 -> ã : ী #! sometimes followed by sp
        "\u00e4": f"{BN.da}{BN.virama}{BN.da}{BN.virama}{BN.ba}",  # \u00e4 -> ä : দ্দ্ব
        "\u00e5": BN.v_u,  # \u00e5 -> å : ু
        "\u00e6": BN.v_u,  # \u00e6 -> æ : ু #! More Below - After combination
        "\u00e7": BN.v_u,  # \u00e7 -> ç :  ু #! On right
        "\u00e8": BN.v_uu,  # \u00e8 -> è :  ূ #! Check when followed by y
        "\u00e9": BN.v_uu,  # \u00e9 -> é : ূ #! NO occurence
        "\u00eb": BN.v_e,  # \u00eb -> ë : ে #! SP
        "\u00ec": BN.v_e,  # \u00ec -> ì : ে #! NOSP
        "\u00ed": BN.v_ai,  # \u00ed -> í : ৈ #! SP
        "\u00ee": BN.v_ai,  # \u00ee -> î : ৈ #! NOSP
        "\u00ef": BN.mark_au,  # \u00ef -> ï : ৗ
        "\u00f0": f"{BN.ja}{BN.virama}{BN.ja}",  # \u00f0 -> ð :জ্জ
        "\u00f1": f"{BN.virama}{BN.ta}{BN.virama}{BN.ta}",  # \u00f1 -> ñ : ্ত্ত #! Down as [st-t]
        "\u00f2": f"\u00a2{BN.virama}",  # \u00f2 -> ò : ্র #! MISUSED (\u0981 -> ঁ )
        "\u00f3": BN.pha,  # \u00f3 -> ó : ফ
        "\u00f4": BN.virama,  # \u00f4 -> ô : ্
        "\u00f5": BN.v_r_vocalic,  # \u00f5 -> õ : ৃ
        "\u00f6": f"{BN.virama}{BN.ra}",  # \u00f6 -> ö : ্র #! Down as [str, pr, dr, ptr]
        "\u00f7": f"{BN.virama}{BN.ra}",  # \u00f7 -> ÷ : ্র #! Down as [mr, sr]
        "\u00f8": f"{BN.virama}{BN.ra}",  # \u00f8 -> ø : ্র #! Down as [pr, khr, gr, dr, shr, br, fr]
        "\u00f9": f"{BN.virama}{BN.ra}",  # \u00f9 -> ù : ্র #! Down as []
        "\u00fa": " ",  # \u00fa -> ú : \u200c -> #! Actually cheikhei
        "\u0152": f"{BN.dha}{BN.virama}{BN.ma}",  # \u0152 -> Œ :ধ্ম
        "\u0153": f"{BN.pa}{BN.virama}{BN.ta}",  # \u0153 -> œ : প্ত
        "\u0160": f"{BN.da}{BN.virama}",  # \u0160 -> Š : দ্ #! Up as []
        "\u0161": BN.pa,  # \u0022 -> š : প
        "\u0178": f"{BN.pa}{BN.virama}",  # \u0178 -> Ÿ : প্ #! Left as []
        "\u0192": BN.da,  # \u0192 -> ƒ : দ
        "\u02c6": f"{BN.da}{BN.virama}{BN.ma}",  # \u02c6 -> ˆ : দ্ম
        "\u02dc": BN.r_vocalic,  # \u02dc -> ˜ : ঋ #! Unused
        "\u2013": f"{BN.na}{BN.virama}",  # \u2013 -> – : ন্ #! Left as [nd, nm, nt]
        "\u2014": f"{BN.virama}{BN.na}",  # \u2014 -> — : ্ন #! Left-Down as [kn, pn, mn, gn, tn]
        "\u201a": f"{BN.virama}{BN.tha}",  # \u201a -> ‚ : ্থ #! Down as [nth, sth]
        "\u201c": f"{BN.na}{BN.virama}{BN.dda}",  # \u201c -> “ : ন্ড
        "\u201d": f"{BN.na}{BN.virama}",  # \u201d -> ” : ন্ #! Up as [nt, nth, ntr, ndr]
        "\u201e": f"{BN.da}{BN.virama}{BN.da}",  # \u201e -> „ : দ্দ
        "\u2020": f"{BN.da}{BN.virama}{BN.dha}{BN.virama}{BN.ba}",  # \u2020 -> † : দ্ধ্ব
        "\u2021": f"{BN.da}{BN.virama}{BN.ba}",  # \u2021 -> ‡ : দ্ব
        "\u2021\u00fd": f"{BN.da}{BN.virama}{BN.dha}",  # \u2021\u00fd -> ‡ý : দ্ধ
        "\u2022": f"{BN.virama}{BN.na}",  # \u2022 -> • : ্ন #! Down as [sn]
        "\u2026": f"{BN.v_ii}{BN.candrabindu}",  # \u2026 -> … : ীঁ
        "\u2030": f"{BN.da}{BN.virama}{BN.ra}",  # \u2030 -> ‰ : দ্র
        "\u2039": BN.dha,  # \u2039 -> ‹ : ধ
        "\u203a": f"{BN.pa}{BN.virama}{BN.sa}",  # \u203a -> › : প্স
        "\u2122": BN.ya,  # \u2122 -> ™ : য
        "\u007d": BN.anusvara,  # \u007d -> } : ং
    }

    R_char_r: Dict[str, str] = {
        "\u00a2": f"{BN.ra}{BN.virama}",  # \u00a2 -> ¢ : BN.ra{BN.virama} -> র্
    }

    s550_extra_chars: Set[str] = {
        "\u00a1",  #! \u00a1 -> ¡ : as joiner or space
        "\u00fc",  #! \u00fc -> ü : as top part of e or u
        "\u00ff",  #! \u00ff -> ÿ
        "\uf000",  #! \uf000 -> 
        "\u0040",  #! \u0040 -> @
        "\u003a",  #! -> :
        BN.visarga,
    }
