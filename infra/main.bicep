targetScope = 'subscription'

@minLength(1)
@maxLength(64)
@description('Name of the environment that can be used as part of naming resource convention')
param environmentName string

@minLength(1)
@description('Primary location for all resources')
param location string

@description('Id of the user or app to assign application roles')
param principalId string

// Optional parameters
@description('SKU for the Container App Environment')
param containerAppsEnvironmentSku string = 'Consumption'

@description('SKU for the Azure Container Registry')
param containerRegistrySku string = 'Basic'

@description('SKU for the PostgreSQL server')
param postgresqlSku string = 'Standard_B1ms'

@description('SKU for the Redis cache')
param redisSku string = 'Basic'

// Generate a unique token to be used in naming resources
var resourceToken = toLower(uniqueString(subscription().id, environmentName, location))

// Organize resources in a resource group
resource rg 'Microsoft.Resources/resourceGroups@2022-09-01' = {
  name: 'rg-${environmentName}'
  location: location
  tags: {
    'azd-env-name': environmentName
  }
}

// Container Apps Environment for hosting the federated learning API
module containerApps 'core/host/container-apps.bicep' = {
  name: 'container-apps'
  scope: rg
  params: {
    name: 'cae-${resourceToken}'
    location: location
    sku: containerAppsEnvironmentSku
    logAnalyticsWorkspaceName: monitoring.outputs.logAnalyticsWorkspaceName
  }
}

// Container Registry for storing application images
module containerRegistry 'core/host/container-registry.bicep' = {
  name: 'container-registry'
  scope: rg
  params: {
    name: 'cr${resourceToken}'
    location: location
    sku: containerRegistrySku
    principalId: principalId
  }
}

// PostgreSQL database for storing federated learning metrics and results
module postgresql 'core/database/postgresql.bicep' = {
  name: 'postgresql'
  scope: rg
  params: {
    name: 'psql-${resourceToken}'
    location: location
    sku: postgresqlSku
    principalId: principalId
    databaseName: 'federatedlearning'
  }
}

// Redis cache for storing temporary federated learning state
module redis 'core/database/redis.bicep' = {
  name: 'redis'
  scope: rg
  params: {
    name: 'redis-${resourceToken}'
    location: location
    sku: redisSku
  }
}

// Storage account for storing model checkpoints and training data
module storage 'core/storage/storage-account.bicep' = {
  name: 'storage'
  scope: rg
  params: {
    name: 'st${resourceToken}'
    location: location
    principalId: principalId
  }
}

// Key Vault for storing secrets and connection strings
module keyVault 'core/security/keyvault.bicep' = {
  name: 'keyvault'
  scope: rg
  params: {
    name: 'kv-${resourceToken}'
    location: location
    principalId: principalId
  }
}

// Monitoring resources (Log Analytics, Application Insights)
module monitoring 'core/monitor/monitoring.bicep' = {
  name: 'monitoring'
  scope: rg
  params: {
    location: location
    logAnalyticsName: 'log-${resourceToken}'
    applicationInsightsName: 'appi-${resourceToken}'
  }
}

// Cognitive Services for ML workloads (optional)
module cognitiveServices 'core/ai/cognitive-services.bicep' = {
  name: 'cognitive-services'
  scope: rg
  params: {
    name: 'cog-${resourceToken}'
    location: location
    principalId: principalId
  }
}

// The main federated learning API container app
module api 'app/api.bicep' = {
  name: 'api'
  scope: rg
  params: {
    name: 'api-${resourceToken}'
    location: location
    imageName: 'federated-learning-api'
    containerAppsEnvironmentName: containerApps.outputs.name
    containerRegistryName: containerRegistry.outputs.name
    postgresqlConnectionString: postgresql.outputs.connectionString
    redisConnectionString: redis.outputs.connectionString
    storageConnectionString: 'DefaultEndpointsProtocol=https;AccountName=${storage.outputs.accountName};EndpointSuffix=${environment().suffixes.storage}'
    applicationInsightsConnectionString: monitoring.outputs.applicationInsightsConnectionString
    keyVaultName: keyVault.outputs.name
  }
}

// Outputs
output AZURE_LOCATION string = location
output AZURE_TENANT_ID string = tenant().tenantId
output AZURE_RESOURCE_GROUP string = rg.name

output AZURE_CONTAINER_REGISTRY_ENDPOINT string = containerRegistry.outputs.loginServer
output AZURE_CONTAINER_REGISTRY_NAME string = containerRegistry.outputs.name

output AZURE_CONTAINER_APPS_ENVIRONMENT_NAME string = containerApps.outputs.name
output AZURE_CONTAINER_APPS_ENVIRONMENT_DEFAULT_DOMAIN string = containerApps.outputs.defaultDomain

output API_BASE_URL string = api.outputs.uri
output API_NAME string = api.outputs.name

output AZURE_POSTGRESQL_CONNECTION_STRING string = postgresql.outputs.connectionString
output AZURE_REDIS_CONNECTION_STRING string = redis.outputs.connectionString
output AZURE_STORAGE_ACCOUNT_NAME string = storage.outputs.accountName

output AZURE_KEY_VAULT_NAME string = keyVault.outputs.name
output AZURE_KEY_VAULT_ENDPOINT string = keyVault.outputs.endpoint

output AZURE_LOG_ANALYTICS_WORKSPACE_NAME string = monitoring.outputs.logAnalyticsWorkspaceName
output AZURE_APPLICATION_INSIGHTS_NAME string = monitoring.outputs.applicationInsightsName
output AZURE_APPLICATION_INSIGHTS_CONNECTION_STRING string = monitoring.outputs.applicationInsightsConnectionString

output AZURE_COGNITIVE_SERVICES_ENDPOINT string = cognitiveServices.outputs.endpoint
output AZURE_COGNITIVE_SERVICES_NAME string = cognitiveServices.outputs.name
