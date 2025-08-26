@description('The name of the container app')
param name string

@description('The location for the container app')
param location string = resourceGroup().location

@description('The name of the container image')
param imageName string

@description('The name of the container apps environment')
param containerAppsEnvironmentName string

@description('The name of the container registry')
param containerRegistryName string

@description('The PostgreSQL connection string')
param postgresqlConnectionString string

@description('The Redis connection string')
param redisConnectionString string

@description('The storage connection string')
param storageConnectionString string

@description('The Application Insights connection string')
param applicationInsightsConnectionString string

@description('The Key Vault name')
param keyVaultName string

resource containerAppsEnvironment 'Microsoft.App/managedEnvironments@2023-05-01' existing = {
  name: containerAppsEnvironmentName
}

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: containerRegistryName
}

resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: name
  location: location
  properties: {
    managedEnvironmentId: containerAppsEnvironment.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        allowInsecure: false
        traffic: [
          {
            latestRevision: true
            weight: 100
          }
        ]
      }
      registries: [
        {
          server: containerRegistry.properties.loginServer
          identity: 'system'
        }
      ]
    }
    template: {
      containers: [
        {
          name: imageName
          image: '${containerRegistry.properties.loginServer}/${imageName}:latest'
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
          env: [
            {
              name: 'POSTGRESQL_CONNECTION_STRING'
              value: postgresqlConnectionString
            }
            {
              name: 'REDIS_CONNECTION_STRING'
              value: redisConnectionString
            }
            {
              name: 'STORAGE_CONNECTION_STRING'
              value: storageConnectionString
            }
            {
              name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
              value: applicationInsightsConnectionString
            }
            {
              name: 'KEY_VAULT_NAME'
              value: keyVaultName
            }
            {
              name: 'PYTHONUNBUFFERED'
              value: '1'
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 10
        rules: [
          {
            name: 'http-requests'
            http: {
              metadata: {
                concurrentRequests: '30'
              }
            }
          }
        ]
      }
    }
  }
  identity: {
    type: 'SystemAssigned'
  }
}

@description('The name of the container app')
output name string = containerApp.name

@description('The FQDN of the container app')
output uri string = 'https://${containerApp.properties.configuration.ingress.fqdn}'

@description('The system-assigned identity principal ID')
output identityPrincipalId string = containerApp.identity.principalId
