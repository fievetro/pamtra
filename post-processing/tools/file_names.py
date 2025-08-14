'''
This file is used to store the paths the different scripts are using, enabling an easy change if necessary.
'''

icon_path_orc = "/work/mh0492/m301067/ec-curtains/data/curtains/build-master-031224/" # Path to Romain's curtains (ICON orcestra version)
icon_path_hack = "/work/mh0492/m301067/ec-curtains/data/curtains/build-hackaton/" # Path to Romain's curtains (ICON hackathon version)
icon_path_new = "/work/mh0492/m301067/ec-curtains/data/curtains/build-new-rain/" # Path to Romain's curtains (ICON new rain version)
earthcare_path = "/work/mh0731/m301196/ecomip/ftp.eorc.jaxa.jp/CPR/2A/CPR_CLP/vBb/" # Path to EarthCARE L2 products by JAXA, downloaded from their ftp server.
halo_path = "ipfs://bafybeigmd3dovwm45ylfqxnn2jphsrdjl2jt3dfytv7grkyhleaq42jthe" # Path to HALO zarr, gotten from the Orcestra data browser.
pamtra_path = "/work/mh0731/m301196/ecomip/pamtra-curtains-july/pamtra-" # Path to the place where I store pamtra curtains made via the script pamtra-orcestra.
jsim_path_orc = "/work/mh0731/m301196/ecomip/jsim/icon_2024-09-03_16_08_00.EASE_ORCESTRA.nc"
jsim_path_hack = "/work/mh0731/m301196/ecomip/jsim/icon_2024-09-03_16_08_00.EASE_HK25.nc"

files_earthcare = ["2024/08/11/ECA_J_CPR_CLP_2AS_20240811T1548_20240811T1559_01162E_vBb.h5", "2024/08/13/ECA_J_CPR_CLP_2AS_20240813T1538_20240813T1550_01193E_vBb.h5", 
                   "2024/08/16/ECA_J_CPR_CLP_2AS_20240816T1610_20240816T1621_01240E_vBb.h5", "2024/08/18/ECA_J_CPR_CLP_2AS_20240818T1559_20240818T1611_01271E_vBb.h5",
                   "2024/08/22/ECA_J_CPR_CLP_2AS_20240822T1539_20240822T1550_01333E_vBb.h5", "2024/08/25/ECA_J_CPR_CLP_2AS_20240825T1609_20240825T1620_01380E_vBb.h5",
                   "2024/08/27/ECA_J_CPR_CLP_2AS_20240827T1558_20240827T1609_01411E_vBb.h5", "2024/08/31/ECA_J_CPR_CLP_2AS_20240831T1536_20240831T1547_01473E_vBb.h5",
                   "2024/09/03/ECA_J_CPR_CLP_2AS_20240903T1605_20240903T1616_01520E_vBb.h5", "2024/09/07/ECA_J_CPR_CLP_2AS_20240907T1542_20240907T1554_01582E_vBb.h5",
                   "2024/09/09/ECA_J_CPR_CLP_2AS_20240909T1703_20240909T1715_01614E_vBb.h5", "2024/09/14/ECA_J_CPR_CLP_2AS_20240914T1722_20240914T1733_01692E_vBb.h5",
                   "2024/09/16/ECA_J_CPR_CLP_2AS_20240916T1711_20240916T1723_01723E_vBb.h5", "2024/09/19/ECA_J_CPR_CLP_2AS_20240919T1741_20240919T1753_01770E_vBb.h5",
                   "2024/09/23/ECA_J_CPR_CLP_2AS_20240923T1719_20240923T1731_01832E_vBb.h5", "2024/09/24/ECA_J_CPR_CLP_2AS_20240924T1800_20240924T1811_01848E_vBb.h5",
                   "2024/09/26/ECA_J_CPR_CLP_2AS_20240926T1749_20240926T1801_01879E_vBb.h5", "2024/09/28/ECA_J_CPR_CLP_2AS_20240928T1738_20240928T1750_01910E_vBb.h5"] # Default names gotten from JAXA ftp server.

files_pamtra = ["2024-08-11_15h51", "2024-08-13_15h39", "2024-08-16_16h13",
                "2024-08-18_16h03", "2024-08-22_15h41", "2024-08-25_16h11",
                "2024-08-27_16h01", "2024-08-31_15h38", "2024-09-03_16h08",
                "2024-09-07_15h47", "2024-09-09_17h05", "2024-09-14_17h25",
                "2024-09-16_17h15", "2024-09-19_17h43", "2024-09-23_17h22",
                "2024-09-24_18h02", "2024-09-26_17h51", "2024-09-28_17h41"]

files_icon = ["orcestra_ec-curtain_2024-08-11_15h51.nc", "orcestra_ec-curtain_2024-08-13_15h39.nc", "orcestra_ec-curtain_2024-08-16_16h13.nc",
              "orcestra_ec-curtain_2024-08-18_16h03.nc", "orcestra_ec-curtain_2024-08-22_15h41.nc", "orcestra_ec-curtain_2024-08-25_16h11.nc",
              "orcestra_ec-curtain_2024-08-27_16h01.nc", "orcestra_ec-curtain_2024-08-31_15h38.nc", "orcestra_ec-curtain_2024-09-03_16h08.nc",
              "orcestra_ec-curtain_2024-09-07_15h47.nc", "orcestra_ec-curtain_2024-09-09_17h05.nc", "orcestra_ec-curtain_2024-09-14_17h25.nc",
              "orcestra_ec-curtain_2024-09-16_17h15.nc", "orcestra_ec-curtain_2024-09-19_17h43.nc", "orcestra_ec-curtain_2024-09-23_17h22.nc",
              "orcestra_ec-curtain_2024-09-24_18h02.nc", "orcestra_ec-curtain_2024-09-26_17h51.nc", "orcestra_ec-curtain_2024-09-28_17h41.nc"]