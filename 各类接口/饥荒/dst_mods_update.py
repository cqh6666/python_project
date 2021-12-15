# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     dst_mods_update
   Description :
   Author :       cqh
   date：          2021/9/6 21:10
-------------------------------------------------
   Change Activity:
                   2021/9/6:
-------------------------------------------------
"""
__author__ = 'cqh'

import re

MOD_SETUP = "./Steam/steamapps/common/Don't Starve Together Dedicated Server/mods/dedicated_server_mods_setup.lua"
MOD_OVERRIDES = ".klei/DoNotStarveCaves_4/Cluster_1/Master/modoverrides.lua"
BACKUP_SETUP = "./Steam/steamapps/common/Don't Starve Together Dedicated " \
               "Server/mods/dedicated_server_mods_setup_backup.lua "


def get_mods_ids():
    """
    得到mods id列表
    :return:
    """
    with open(MOD_OVERRIDES, 'r', encoding="utf-8") as f:
        content = f.read()
        pattern = re.compile(r'(?<=workshop-).*?(?="])')
        ids = pattern.findall(content)
        print("检测到有{}个mod,正转化为指令并插入setup脚本...".format(len(ids)))
        return ids


def format_mods(ids):
    """
    将id列表转化为特定格式
    :param ids:
    :return:
    """
    smt = "ServerModSetup"
    f_ids = [smt + '("' + mid + '")' for mid in ids]
    return f_ids


def move_to_setup(f_ids):
    """
    将特定格式写入到mod_setup文件
    :return:
    """
    with open(MOD_SETUP, "w") as f:
        for fid in f_ids:
            f.write(fid + "\n")
        print("已将指令插入到setup脚本中...")


def backup_file(source, target):
    """
    文件备份
    :return:
    """
    updated = True
    with open(source, 'r') as f:
        if "ServerModCollectionSetup" in f.readline():
            updated = False

    with open(source, 'rb') as f1:
        if not updated:
            with open(target, 'wb') as f2:
                content = f1.read()
                f2.write(content)
                print("已备份setup文件...")


if __name__ == '__main__':
    ids = get_mods_ids()
    # f_ids = format_mods(ids)
    # backup_file(MOD_SETUP, BACKUP_SETUP)
    # move_to_setup(f_ids)
