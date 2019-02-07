#
_XDCBUILDCOUNT = 
ifneq (,$(findstring path,$(_USEXDCENV_)))
override XDCPATH = /Applications/ti/tirtos_cc13xx_cc26xx_2_21_00_06/packages;/Applications/ti/tirtos_cc13xx_cc26xx_2_21_00_06/products/tidrivers_cc13xx_cc26xx_2_21_00_04/packages;/Applications/ti/tirtos_cc13xx_cc26xx_2_21_00_06/products/bios_6_46_01_37/packages;/Applications/ti/tirtos_cc13xx_cc26xx_2_21_00_06/products/uia_2_01_00_01/packages;/Users/sseeberger/Documents/Github/NEU/eece5155-WirelessIoT/pwmled_CC2650_LAUNCHXL_TI/.config
override XDCROOT = /Applications/ti/xdctools_3_32_02_25_core
override XDCBUILDCFG = ./config.bld
endif
ifneq (,$(findstring args,$(_USEXDCENV_)))
override XDCARGS = 
override XDCTARGETS = 
endif
#
ifeq (0,1)
PKGPATH = /Applications/ti/tirtos_cc13xx_cc26xx_2_21_00_06/packages;/Applications/ti/tirtos_cc13xx_cc26xx_2_21_00_06/products/tidrivers_cc13xx_cc26xx_2_21_00_04/packages;/Applications/ti/tirtos_cc13xx_cc26xx_2_21_00_06/products/bios_6_46_01_37/packages;/Applications/ti/tirtos_cc13xx_cc26xx_2_21_00_06/products/uia_2_01_00_01/packages;/Users/sseeberger/Documents/Github/NEU/eece5155-WirelessIoT/pwmled_CC2650_LAUNCHXL_TI/.config;/Applications/ti/xdctools_3_32_02_25_core/packages;..
HOSTOS = MacOS
endif
