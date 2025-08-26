@description('The name of the storage account')
param name string

@description('The location for the storage account')
param location string = resourceGroup().location

@description('The principal ID to assign roles to')
param principalId string

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: name
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: true
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    networkAcls: {
      defaultAction: 'Allow'
    }
  }
}

// Create blob container for model checkpoints
resource blobContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  name: '${storageAccount.name}/default/models'
  properties: {
    publicAccess: 'None'
  }
}

// Create blob container for training data
resource dataContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  name: '${storageAccount.name}/default/data'
  properties: {
    publicAccess: 'None'
  }
}

// Assign Storage Blob Data Contributor role to the principal
resource storageRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: storageAccount
  name: guid(storageAccount.id, principalId, 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe') // Storage Blob Data Contributor
    principalId: principalId
    principalType: 'User'
  }
}

@description('The name of the storage account')
output name string = storageAccount.name

@description('The storage account name for connection string construction')
output accountName string = storageAccount.name

@description('The primary blob endpoint for the storage account')
output primaryBlobEndpoint string = storageAccount.properties.primaryEndpoints.blob

@description('The resource ID of the storage account')
output resourceId string = storageAccount.id
