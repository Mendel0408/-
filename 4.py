# insert polygon and points + features 插入多边形和点及相关特征点数据到GIS中
# insert polygon to buidling GIS!!! 插入多边形以构建GIS
workspace = arcpy.env.workspace

# Create a feature class with a spatial reference of GCS WGS 1984 创建一个空间参考为GCS WGS 1984的要素类
result = arcpy.management.CreateFeatureclass(
    workspace, "building_area_1", "polygon", has_z="ENABLED",
    spatial_reference=2039)
feature_class = result[0]

# Write feature to new feature class 将多边形要素写入新的要素类
with arcpy.da.InsertCursor(feature_class, ['SHAPE@']) as cursor:
    cursor.insertRow([res77])  # res77

# 创建一个新的点要素类
new_shape_file = arcpy.CreateFeatureclass_management(r"C:\Users\liat\pythonGis", "features_1.shp", "POINT",
                                                     has_z="ENABLED", spatial_reference=2039)
print(new_shape_file)
# 向新的点要素类添加一个字段
arcpy.AddField_management(new_shape_file, "NAME", "TEXT")

# 将点要素插入到新的点要素类中
with arcpy.da.InsertCursor(new_shape_file, ['NAME', 'SHAPE@']) as insert_cursor:
    for coord in calc_pts:
        print("Inserted {} into {}".format(coord, new_shape_file))
        insert_cursor.insertRow(coord)