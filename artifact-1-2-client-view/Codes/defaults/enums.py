from enum import Enum

class WiFi24GHZChannels(Enum):
    CHANNEL_1   = 2412
    CHANNEL_6   = 2437
    CHANNEL_11  = 2462    

class WiFi24GHZChannelsNo(Enum):
    CHANNEL_1 = 1
    CHANNEL_6 = 6
    CHANNEL_11 = 11

class WiFiOperatingClass(Enum):
    # 2.4 GHz band
    BAND_2_4GHz_20MHz = 81  # 2.4GHz 20MHz (channels 1–13)
    BAND_2_4GHz_40MHz = 82  # 2.4GHz 40MHz (channels 3–11)
    BAND_2_4GHz_20MHz_CH14 = 83  # 2.4GHz 20MHz (channel 14)
    
    # 5 GHz band
    BAND_5GHz_20MHz_LOW = 115  # 5GHz 20MHz (36–48)
    BAND_5GHz_40MHz_LOW = 116  # 5GHz 40MHz (38–46)
    BAND_5GHz_80MHz_LOW = 117  # 5GHz 80MHz (42)
    BAND_5GHz_160MHz_LOW = 118  # 5GHz 160MHz (50)
    BAND_5GHz_20MHz_HIGH = 119  # 5GHz 20MHz (52–64)
    BAND_5GHz_40MHz_HIGH = 120  # 5GHz 40MHz (54–62)
    BAND_5GHz_80MHz_HIGH = 121  # 5GHz 80MHz (58)
    BAND_5GHz_160MHz_HIGH = 122  # 5GHz 160MHz (50)
    
    # 6 GHz band
    BAND_6GHz_20MHz = 125  # 6GHz 20MHz (1–233)
    BAND_6GHz_40MHz = 126  # 6GHz 40MHz (3–231)
    BAND_6GHz_80MHz = 127  # 6GHz 80MHz (7–227)
    BAND_6GHz_160MHz = 128  # 6GHz 160MHz (15–219)
    BAND_6GHz_20MHz_LOW_BAND = 129  # 6GHz 20MHz (low band)
    BAND_6GHz_40MHz_LOW_BAND = 130  # 6GHz 40MHz (low band)
    BAND_6GHz_80MHz_LOW_BAND = 131  # 6GHz 80MHz (low band)
    BAND_6GHz_160MHz_LOW_BAND = 132  # 6GHz 160MHz (low band)
    
    # 60 GHz band
    BAND_60GHz_CHANNEL_1 = 180  # 60GHz (Channel 1)
    BAND_60GHz_CHANNEL_2 = 181  # 60GHz (Channel 2)
    BAND_60GHz_CHANNEL_3 = 182  # 60GHz (Channel 3)
    BAND_60GHz_CHANNEL_4 = 183  # 60GHz (Channel 4)
 