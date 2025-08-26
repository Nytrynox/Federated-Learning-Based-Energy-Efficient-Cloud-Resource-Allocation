@description('Name of the PostgreSQL server')
param name string

@description('Location for all resources')
param location string = resourceGroup().location

@description('SKU for the PostgreSQL server')
param sku string = 'Standard_B1ms'

@description('Principal ID to assign roles to')
param principalId string

@description('Database name')
param databaseName string = 'federatedlearning'

@description('Administrator login username')
param administratorLogin string = 'pgadmin'

@secure()
@description('Administrator login password')
param administratorPassword string = '${uniqueString(resourceGroup().id)}Aa1!'

resource postgresqlServer 'Microsoft.DBforPostgreSQL/flexibleServers@2023-06-01-preview' = {
  name: name
  location: location
  sku: {
    name: sku
    tier: 'Burstable'
  }
  properties: {
    administratorLogin: administratorLogin
    administratorLoginPassword: administratorPassword
    storage: {
      storageSizeGB: 32
      iops: 120
      autoGrow: 'Enabled'
    }
    backup: {
      backupRetentionDays: 7
      geoRedundantBackup: 'Disabled'
    }
    network: {
      publicNetworkAccess: 'Enabled'
    }
    highAvailability: {
      mode: 'Disabled'
    }
    maintenanceWindow: {
      customWindow: 'Disabled'
      dayOfWeek: 0
      startHour: 0
      startMinute: 0
    }
    version: '15'
  }
}

// Allow Azure services to access the server
resource firewallRule 'Microsoft.DBforPostgreSQL/flexibleServers/firewallRules@2023-06-01-preview' = {
  name: 'AllowAzureServices'
  parent: postgresqlServer
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
}

// Create the database
resource database 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2023-06-01-preview' = {
  name: databaseName
  parent: postgresqlServer
  properties: {
    charset: 'UTF8'
    collation: 'en_US.UTF8'
  }
}

output name string = postgresqlServer.name
output id string = postgresqlServer.id
output fqdn string = postgresqlServer.properties.fullyQualifiedDomainName
output connectionString string = 'postgresql://${administratorLogin}:${administratorPassword}@${postgresqlServer.properties.fullyQualifiedDomainName}:5432/${databaseName}?sslmode=require'
