@description('Name of the Redis cache')
param name string

@description('Location for all resources')
param location string = resourceGroup().location

@description('SKU for the Redis cache')
@allowed(['Basic', 'Standard', 'Premium'])
param sku string = 'Basic'

@description('Redis cache capacity')
param capacity int = 0

resource redisCache 'Microsoft.Cache/redis@2023-08-01' = {
  name: name
  location: location
  properties: {
    sku: {
      name: sku
      family: sku == 'Premium' ? 'P' : 'C'
      capacity: capacity
    }
    enableNonSslPort: false
    minimumTlsVersion: '1.2'
    publicNetworkAccess: 'Enabled'
    redisConfiguration: {
      'maxmemory-reserved': '30'
      'maxfragmentationmemory-reserved': '30'
      'maxmemory-delta': '30'
    }
  }
}

output name string = redisCache.name
output id string = redisCache.id
output hostName string = redisCache.properties.hostName
output sslPort int = redisCache.properties.sslPort
output primaryKey string = redisCache.listKeys().primaryKey
output connectionString string = '${redisCache.properties.hostName}:${redisCache.properties.sslPort},password=${redisCache.listKeys().primaryKey},ssl=True,abortConnect=False'
