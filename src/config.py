
WEIGHING_METHOD = 'ISNS'
N_CLASSES = 5
COLOR_DROPOUT = 0.2

"""     CAT-3 Segmentation labels:
        0 -> ground
        1 -> tower
        2 -> lines
        3 -> vegetation
        4 -> wind_turbine 
        5 -> buildings and roofs
        6 -> cranes
        
        dataset 50x50:
        total                2460542319
        other                     62616
        tower                    276977
        lines                     96623
        low_veg               759421960
        high_veg             1452379415
        wind_turbine              27516
        ground                227933639
        building               20343573


SAMPLES_X_CLASS_CAT3 = [
    62616 + 227933639,  # 5 other
    276977,  # 1 transmission towers
    96623,  # 2 lines
    2211801375,  # 3 veg
    20343573,  # 4 buildings and roofs
    27516 + 276977
]
"""

"""
 CAT3 Segmentation 120x120
    total           7812844111
    no_ground       7812844111
    other               358643
    tower              2737046
    lines              5010824
    low_veg         5142946083
    high_veg        2627200737
    wind_turbine        565938
    building          34024840
    
    no_ground       100.000000
    other             0.004590
    tower             0.035033
    lines             0.064136
    low_veg          65.826810
    high_veg         33.626688
    wind_turbine      0.007244
    building          0.435499
    
    number of files:
    total files: 49512
    paths_towers: 9366
    paths_lines: 3198
    veg: 36145
    paths_wind: 123
    paths_other: 680
"""
SAMPLES_X_CLASS_CAT3 = [
    480,                    # 0 ground
    20,                    # 1 transmission towers abans 2500
    27,                    # 2 lines
    37000                   # 3 veg
    ]

# SAMPLES_X_CLASS_CAT3 = [
#     1340000,                    # 0 ground
#     400,                    # 1 transmission towers abans 2500
#     500,                    # 2 lines
#     1680000,                   # 3 veg
#     600]                    # wind turbine 46k
# [0.1022, 0.3098, 0.2596, 0.1022, 0.2264]

# -------------------------------------------------- DALES --------------------------------------------------
SAMPLES_X_CLASS_DALES = [
    186 * 10 ** 6,  #  other
    178 * 10 ** 6,  #  ground
    0.28 * 10 ** 6, #  poles
    0.8 * 10 ** 6,  #  power lines
]

SAMPLES_X_CLASS_DALES_ALL = [
    178 * 10 ** 6,  #  ground
    1 * 10 ** 6,  #  poles
    1 * 10 ** 6, #  power lines
    121 * 10 ** 6,  #  vegetation
    58 * 10 ** 6,    #  buildings
    1 * 10 ** 6,    # cars and trucks
]

# SAMPLES_X_CLASS_DALES = [
#     178 * 10 ** 6,  #  ground
#     0.8 * 10 ** 6,  #  power lines
#     0.28 * 10 ** 6, #  poles
#     121 * 10 ** 6,  #  vegetation
#     2 * 10 ** 6,    #  fences
#     57 * 10 ** 6,   #  buildings
#     6 * 10 ** 6,    # cars
# ]