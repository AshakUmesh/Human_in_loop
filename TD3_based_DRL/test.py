import carla

client = carla.Client("localhost", 2000)
client.set_timeout(240.0)

world = client.get_world()
print("Connected to:", world.get_map().name)